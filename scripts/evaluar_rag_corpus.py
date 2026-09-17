#!/usr/bin/env python3
"""Evalúa embeddings y chunks de un corpus RAG.

Genera:
- resumen_metricas.json
- metricas_agregadas.csv
- chunks_metricas.csv
- vecino_mas_cercano.csv
- duplicados_muestra.json
- retrieval_resultados.csv/json (si se proporciona un JSONL de evaluación)

Formatos aceptados:
embeddings.json: lista de registros con embedding/vector y opcionalmente id,
metadata, text/document/page_content.
chunks.json: lista de registros con id, text/document/page_content y metadata,
o un diccionario con una clave lista: chunks/documents/records/data.

Evaluación retrieval JSONL/JSON:
{
  "query": "...",
  "relevant_ids": ["chunk-id-1", "chunk-id-2"],
  "relevance": {"chunk-id-1": 2, "chunk-id-2": 1},
  "reference_answer": "..."  # opcional; no se usa sin generación
}

Uso:
python evaluar_rag_corpus.py \
  --embeddings output/embeddings.json \
  --chunks output/chunks.json \
  --eval eval_queries.jsonl \
  --out output/evaluacion_rag \
  --k 1 3 5 10

Dependencias:
pip install numpy pandas scikit-learn
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:
    raise SystemExit("Falta scikit-learn: pip install scikit-learn") from exc


TEXT_KEYS = ("text", "document", "page_content", "content", "texto")
VECTOR_KEYS = ("embedding", "vector", "values", "embedding_vector")
ID_KEYS = ("id", "chunk_id", "document_id", "uid")
LIST_KEYS = ("chunks", "documents", "records", "data", "items")


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Carga JSON lista/dict o JSONL."""
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if line.strip():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSONL inválido en línea {line_no}: {exc}") from exc
                    if not isinstance(value, dict):
                        raise ValueError(f"La línea {line_no} no contiene un objeto JSON")
                    rows.append(value)
        return rows

    obj = json_load(path)
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        rows = None
        for key in LIST_KEYS:
            if isinstance(obj.get(key), list):
                rows = obj[key]
                break
        if rows is None:
            rows = [obj]
    else:
        raise ValueError(f"Formato no admitido en {path}: se esperaba lista/dict")

    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} debe contener objetos JSON")
    return rows


def first_value(row: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def metadata_of(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata", row.get("meta", {}))
    return value if isinstance(value, dict) else {"metadata_raw": str(value)}


def text_of(row: dict[str, Any]) -> str:
    value = first_value(row, TEXT_KEYS, "")
    return "" if value is None else str(value)


def id_of(row: dict[str, Any], index: int, metadata: dict[str, Any]) -> str:
    value = first_value(row, ID_KEYS)
    if value is None:
        value = first_value(metadata, ID_KEYS)
    return str(value) if value is not None else f"row_{index:08d}"


def vector_of(row: dict[str, Any]) -> list[float] | None:
    value = first_value(row, VECTOR_KEYS)
    if value is None and isinstance(row.get("data"), dict):
        value = first_value(row["data"], VECTOR_KEYS)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("Un embedding no es una lista")
    return [float(x) for x in value]


def normalize_records(rows: list[dict[str, Any]], with_vectors: bool) -> list[dict[str, Any]]:
    out = []
    for index, row in enumerate(rows):
        metadata = metadata_of(row)
        normalized = {
            "id": id_of(row, index, metadata),
            "text": text_of(row),
            "metadata": metadata,
            "source": metadata.get("source", metadata.get("fuente", "")),
            "document_id": metadata.get("document_id", metadata.get("doc_id", "")),
            "category": metadata.get("doc_category", metadata.get("category", metadata.get("categoria", ""))),
        }
        if with_vectors:
            normalized["vector"] = vector_of(row)
        out.append(normalized)
    return out


def flatten_metadata(record: dict[str, Any]) -> dict[str, Any]:
    values = dict(record["metadata"])
    values.update({
        "id": record["id"],
        "source": record["source"],
        "document_id": record["document_id"],
        "category": record["category"],
    })
    return values


def safe_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe_json(value), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def validate_embeddings(records: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    valid = []
    missing = []
    dimensions = []
    for record in records:
        vector = record.get("vector")
        if vector is None:
            missing.append(record["id"])
            continue
        dimensions.append(len(vector))
        valid.append(record)

    if not valid:
        raise ValueError("No hay embeddings válidos en embeddings.json")

    if len(set(dimensions)) != 1:
        raise ValueError(f"Dimensiones inconsistentes: {Counter(dimensions)}")

    matrix = np.asarray([r["vector"] for r in valid], dtype=np.float32)
    finite_rows = np.isfinite(matrix).all(axis=1)
    norms = np.linalg.norm(np.nan_to_num(matrix), axis=1)
    zero_rows = norms == 0
    ids = [r["id"] for r in valid]
    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
    metadata_missing = sum(not bool(r["metadata"]) for r in valid)
    metadata_without_source = sum(not bool(r["source"]) for r in valid)
    texts_empty = sum(not r["text"].strip() for r in valid)

    metrics = {
        "embedding_records_input": len(records),
        "embedding_records_valid": len(valid),
        "embedding_records_without_vector": len(missing),
        "embedding_dimension": matrix.shape[1],
        "dimension_distribution": dict(Counter(dimensions)),
        "invalid_nan_inf_vectors": int((~finite_rows).sum()),
        "zero_vectors": int(zero_rows.sum()),
        "duplicate_ids": len(duplicate_ids),
        "duplicate_id_values_sample": duplicate_ids[:20],
        "metadata_empty": metadata_missing,
        "metadata_without_source": metadata_without_source,
        "text_empty": texts_empty,
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
        "norm_p25": float(np.percentile(norms, 25)),
        "norm_p50": float(np.percentile(norms, 50)),
        "norm_p75": float(np.percentile(norms, 75)),
    }
    return matrix, ids, {"records": valid, "metrics": metrics}


def length_metrics(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lengths = np.asarray([len(r["text"]) for r in records], dtype=np.int64)
    if len(lengths) == 0:
        return {}, []
    categories = Counter(str(r["category"]) or "__sin_categoria__" for r in records)
    source_counts = Counter(str(r["source"]) or "__sin_fuente__" for r in records)
    document_counts = Counter(str(r["document_id"]) or "__sin_documento__" for r in records)
    per_doc = Counter(str(r["document_id"]) or "__sin_documento__" for r in records)
    rows = []
    for record in records:
        rows.append({
            "id": record["id"],
            "length_chars": len(record["text"]),
            "empty": not bool(record["text"].strip()),
            "short_lt_10": len(record["text"].strip()) < 10,
            "short_lt_50": len(record["text"].strip()) < 50,
            "long_gt_2000": len(record["text"]) > 2000,
            "long_gt_3000": len(record["text"]) > 3000,
            "source": record["source"],
            "document_id": record["document_id"],
            "category": record["category"],
        })
    metrics = {
        "chunk_records": len(records),
        "length_min_chars": int(lengths.min()),
        "length_mean_chars": float(lengths.mean()),
        "length_max_chars": int(lengths.max()),
        "length_p10_chars": float(np.percentile(lengths, 10)),
        "length_p25_chars": float(np.percentile(lengths, 25)),
        "length_p50_chars": float(np.percentile(lengths, 50)),
        "length_p75_chars": float(np.percentile(lengths, 75)),
        "length_p90_chars": float(np.percentile(lengths, 90)),
        "length_p95_chars": float(np.percentile(lengths, 95)),
        "length_p99_chars": float(np.percentile(lengths, 99)),
        "empty_chunks": int(sum(not r["text"].strip() for r in records)),
        "chunks_lt_10": int((lengths < 10).sum()),
        "chunks_lt_50": int((lengths < 50).sum()),
        "chunks_gt_2000": int((lengths > 2000).sum()),
        "chunks_gt_3000": int((lengths > 3000).sum()),
        "unique_sources": len(source_counts),
        "unique_documents": len(document_counts),
        "chunks_per_document_mean": float(np.mean(list(per_doc.values()))) if per_doc else 0.0,
        "chunks_per_document_median": float(np.median(list(per_doc.values()))) if per_doc else 0.0,
        "chunks_per_document_max": int(max(per_doc.values())) if per_doc else 0,
        "documents_by_source": dict(source_counts),
        "chunks_by_category": dict(categories),
    }
    return metrics, rows


def cosine_nearest_neighbors(matrix: np.ndarray, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Obtiene vecino más cercano excluyendo el propio, por lotes.

    Los vectores se normalizan para que producto punto = coseno. Se usa
    NearestNeighbors con métrica cosine; no materializa una matriz n x n.
    """
    if len(matrix) <= 1:
        return np.zeros(len(matrix), dtype=np.float32), np.full(len(matrix), -1, dtype=np.int64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(normalized)
    distances, indices = nn.kneighbors(normalized, n_neighbors=2)
    neighbor_pos = np.zeros(len(matrix), dtype=np.int64)
    neighbor_sim = np.zeros(len(matrix), dtype=np.float32)
    for i in range(len(matrix)):
        choices = [(float(d), int(j)) for d, j in zip(distances[i], indices[i]) if int(j) != i]
        if choices:
            distance, position = min(choices)
            neighbor_pos[i] = position
            neighbor_sim[i] = 1.0 - distance
        else:
            neighbor_pos[i] = -1
    return neighbor_sim, neighbor_pos


def redundancy_metrics(records: list[dict[str, Any]], matrix: np.ndarray, ids: list[str], threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    similarities, positions = cosine_nearest_neighbors(matrix)
    rows = []
    pairs = []
    for i, record in enumerate(records):
        j = int(positions[i])
        neighbor_id = ids[j] if j >= 0 else ""
        row = {
            "id": record["id"],
            "nearest_neighbor_id": neighbor_id,
            "nearest_neighbor_cosine": float(similarities[i]),
            "above_dedup_threshold": bool(similarities[i] >= threshold),
            "source": record["source"],
            "document_id": record["document_id"],
            "category": record["category"],
        }
        rows.append(row)
        if j >= 0 and similarities[i] >= threshold and record["id"] != neighbor_id:
            pairs.append({
                "id_a": record["id"],
                "id_b": neighbor_id,
                "cosine": float(similarities[i]),
                "text_a": record["text"],
                "text_b": records[j]["text"],
                "metadata_a": record["metadata"],
                "metadata_b": records[j]["metadata"],
            })
    metrics = {
        "nearest_neighbor_cosine_mean": float(similarities.mean()) if len(similarities) else 0.0,
        "nearest_neighbor_cosine_min": float(similarities.min()) if len(similarities) else 0.0,
        "nearest_neighbor_cosine_max": float(similarities.max()) if len(similarities) else 0.0,
        "nearest_neighbor_cosine_p50": float(np.percentile(similarities, 50)) if len(similarities) else 0.0,
        "nearest_neighbor_cosine_p90": float(np.percentile(similarities, 90)) if len(similarities) else 0.0,
        "nearest_neighbor_cosine_p95": float(np.percentile(similarities, 95)) if len(similarities) else 0.0,
        "nearest_neighbor_cosine_p99": float(np.percentile(similarities, 99)) if len(similarities) else 0.0,
        "chunks_above_dedup_threshold": int((similarities >= threshold).sum()),
        "chunks_above_dedup_threshold_pct": float((similarities >= threshold).mean() * 100) if len(similarities) else 0.0,
    }
    return metrics, rows, pairs


def category_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_category = defaultdict(lambda: {"chunks": 0, "sources": set(), "documents": set(), "lengths": []})
    for record in records:
        key = str(record["category"]) or "__sin_categoria__"
        item = by_category[key]
        item["chunks"] += 1
        item["sources"].add(str(record["source"]))
        item["documents"].add(str(record["document_id"]))
        item["lengths"].append(len(record["text"]))
    result = {}
    total = len(records)
    for category, item in sorted(by_category.items()):
        result[category] = {
            "chunks": item["chunks"],
            "share_pct": item["chunks"] / total * 100 if total else 0.0,
            "sources": len(item["sources"]),
            "documents": len(item["documents"]),
            "mean_length_chars": float(np.mean(item["lengths"])) if item["lengths"] else 0.0,
        }
    return result


def metadata_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = Counter()
    tags = Counter()
    tag_chunks = Counter()
    for record in records:
        keys.update(record["metadata"].keys())
        for key, value in record["metadata"].items():
            if key.startswith("tag_") and value is True:
                tag = key[4:]
                tags[tag] += 1
                tag_chunks[tag] += 1
    return {
        "metadata_key_counts": dict(keys),
        "tag_counts": dict(tags),
        "tag_top3": [[name, count] for name, count in tags.most_common(3)],
        "records_without_category": sum(not bool(r["category"]) for r in records),
        "records_without_source": sum(not bool(r["source"]) for r in records),
        "records_without_document_id": sum(not bool(r["document_id"]) for r in records),
    }


def load_eval(path: Path) -> list[dict[str, Any]]:
    return load_records(path)


def query_embedding(query: str, all_records: list[dict[str, Any]], matrix: np.ndarray) -> np.ndarray:
    """Placeholder deliberado: retrieval offline requiere query_embeddings.json.

    Se puede proporcionar un campo query_embedding en cada caso de evaluación.
    Así se evita generar embeddings con un modelo diferente al usado al indexar.
    """
    raise RuntimeError("No hay query_embedding; proporciona embeddings de las consultas con el mismo modelo")


def retrieval_eval(eval_rows: list[dict[str, Any]], records: list[dict[str, Any]], matrix: np.ndarray, ids: list[str], ks: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    id_to_pos = {value: index for index, value in enumerate(ids)}
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    result_rows = []
    for case_index, case in enumerate(eval_rows):
        query = str(case.get("query", ""))
        query_vector = case.get("query_embedding", case.get("embedding"))
        if query_vector is None:
            raise ValueError(f"Caso {case_index} no contiene query_embedding")
        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != normalized.shape[1]:
            raise ValueError(f"Dimensión incorrecta en query {case_index}: {q.shape}; esperada {normalized.shape[1]}")
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = normalized @ q
        order = np.argsort(-scores)
        relevant = set(map(str, case.get("relevant_ids", case.get("relevant", []))))
        relevance = {str(key): float(value) for key, value in (case.get("relevance") or {}).items()}
        if not relevant:
            relevant = set(relevance)
        row = {"case_index": case_index, "query": query, "relevant_count": len(relevant)}
        ranking = []
        for position, index in enumerate(order, 1):
            ranking.append({"id": ids[int(index)], "score": float(scores[int(index)]), "rank": position})
        first_rank = next((item["rank"] for item in ranking if item["id"] in relevant), None)
        row["rr"] = 1.0 / first_rank if first_rank else 0.0
        for k in ks:
            top_ids = [item["id"] for item in ranking[:k]]
            hits = len(set(top_ids) & relevant)
            row[f"hits@{k}"] = hits
            row[f"recall@{k}"] = hits / len(relevant) if relevant else 0.0
            row[f"precision@{k}"] = hits / k
            gains = [relevance.get(item_id, 1.0 if item_id in relevant else 0.0) for item_id in top_ids]
            dcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, 1))
            ideal = sorted(relevance.values() or [1.0] * len(relevant), reverse=True)[:k]
            idcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(ideal, 1))
            row[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
            row[f"top_score@{k}"] = float(scores[order[:k]].max()) if k else 0.0
        row["top_ids"] = json.dumps([item["id"] for item in ranking[:max(ks)]], ensure_ascii=False)
        result_rows.append(row)

    summary = {"queries": len(result_rows)}
    for k in ks:
        for metric in ("recall", "precision", "ndcg"):
            key = f"{metric}@{k}"
            summary[f"mean_{key}"] = float(np.mean([row[key] for row in result_rows])) if result_rows else 0.0
    summary["mrr"] = float(np.mean([row["rr"] for row in result_rows])) if result_rows else 0.0
    summary["queries_with_hit@1"] = int(sum(row.get("hits@1", 0) > 0 for row in result_rows)) if result_rows else 0
    return result_rows, summary


def aggregate_csv(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for section, values in metrics.items():
        if isinstance(values, dict):
            for metric, value in values.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    rows.append({"section": section, "metric": metric, "value": value})
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evalúa un corpus vectorial RAG desde embeddings.json y chunks.json")
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--eval", type=Path, default=None, help="JSONL/JSON con query_embedding y relevant_ids")
    parser.add_argument("--out", type=Path, default=Path("output/evaluacion_rag"))
    parser.add_argument("--dedup-threshold", type=float, default=0.93)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--sample-pairs", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if any(k <= 0 for k in args.k):
        raise SystemExit("Todos los valores de --k deben ser positivos")
    args.out.mkdir(parents=True, exist_ok=True)

    embedding_rows = normalize_records(load_records(args.embeddings), with_vectors=True)
    chunk_rows = normalize_records(load_records(args.chunks), with_vectors=False)
    matrix, embedding_ids, embedding_payload = validate_embeddings(embedding_rows)
    valid_embedding_records = embedding_payload["records"]

    chunk_by_id = {row["id"]: row for row in chunk_rows}
    aligned_records = []
    for record in valid_embedding_records:
        aligned = dict(record)
        chunk = chunk_by_id.get(record["id"])
        if chunk:
            aligned["text"] = chunk["text"] or record["text"]
            aligned["metadata"] = {**chunk["metadata"], **record["metadata"]}
            aligned["source"] = chunk["source"] or record["source"]
            aligned["document_id"] = chunk["document_id"] or record["document_id"]
            aligned["category"] = chunk["category"] or record["category"]
        aligned_records.append(aligned)

    matrix, embedding_ids, embedding_payload = validate_embeddings(aligned_records)
    valid_embedding_records = embedding_payload["records"]
    length_summary, length_rows = length_metrics(valid_embedding_records)
    redundancy_summary, neighbor_rows, duplicate_pairs = redundancy_metrics(valid_embedding_records, matrix, embedding_ids, args.dedup_threshold)
    metadata_summary = metadata_metrics(valid_embedding_records)
    categories = category_metrics(valid_embedding_records)

    summary: dict[str, Any] = {
        "input": {
            "embeddings": str(args.embeddings),
            "chunks": str(args.chunks),
            "evaluation": str(args.eval) if args.eval else None,
            "output": str(args.out),
        },
        "integrity": embedding_payload["metrics"],
        "chunk_quality": length_summary,
        "metadata": metadata_summary,
        "categories": categories,
        "redundancy": redundancy_summary,
        "dedup_configuration": {
            "threshold": args.dedup_threshold,
            "above_threshold_meaning": "candidato potencial a deduplicación; requiere validación de texto/metadatos",
        },
    }

    write_json(args.out / "resumen_metricas.json", summary)
    write_csv(args.out / "metricas_agregadas.csv", aggregate_csv(summary))
    write_csv(args.out / "chunks_metricas.csv", length_rows)
    write_csv(args.out / "vecino_mas_cercano.csv", neighbor_rows)
    write_json(args.out / "duplicados_muestra.json", duplicate_pairs[:args.sample_pairs])

    if args.eval:
        eval_rows = load_eval(args.eval)
        retrieval_rows, retrieval_summary = retrieval_eval(eval_rows, valid_embedding_records, matrix, embedding_ids, sorted(set(args.k)))
        write_csv(args.out / "retrieval_resultados.csv", retrieval_rows)
        write_json(args.out / "retrieval_resultados.json", {"summary": retrieval_summary, "cases": retrieval_rows})
        summary["retrieval"] = retrieval_summary
        write_json(args.out / "resumen_metricas.json", summary)
        write_csv(args.out / "metricas_agregadas.csv", aggregate_csv(summary))

    print(f"Resultados escritos en: {args.out}")
    print(f"Embeddings válidos: {len(valid_embedding_records)}; dimensión: {matrix.shape[1]}")
    print(f"Redundancia media al vecino más cercano: {redundancy_summary['nearest_neighbor_cosine_mean']:.4f}")
    print(f"Candidatos sobre umbral {args.dedup_threshold:.2f}: {redundancy_summary['chunks_above_dedup_threshold']}")
    if args.eval:
        print(f"MRR: {summary['retrieval']['mrr']:.4f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
