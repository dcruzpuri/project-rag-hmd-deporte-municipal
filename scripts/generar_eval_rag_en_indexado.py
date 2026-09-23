"""
Genera un gold set JSONL durante el indexado del corpus.

Estrategia:
1. Selecciona chunks reales del corpus ya cargado/chunkeado.
2. Pide al LLM preguntas y respuestas fundamentadas en esos chunks.
3. Para cada pregunta, obtiene query_embedding con el mismo modelo del índice.
4. Resuelve relevant_ids con una búsqueda vectorial contra TODOS los chunks.
5. Conserva como relevantes los chunks fuente de la pregunta y, opcionalmente,
   los vecinos que el LLM confirma como necesarios/parciales.

Integración sugerida desde el pipeline:
    eval_records = generar_eval_set(
        chunks=chunks_post_tsd,
        embedder=embeddear,
        llm_call=llm_call,
        n=50,
        output_path="output/eval_queries.jsonl",
    )

El contrato de llm_call es:
    llm_call(prompt: str) -> str

La respuesta del LLM debe ser JSON con esta forma:
{
  "query": "...",
  "reference_answer": "...",
  "relevance": {
      "id_chunk_fuente_01": 10},
      "id_chunk_fuente_23": 3},
  "question_type": "factual"
}

Para consultas sin respuesta, `relevance` debe ser {} y la respuesta debe
indicar claramente que el corpus no contiene información suficiente.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

QUESTION_TYPES = {
    "factual": 10,
    "entity": 7,
    "comparison": 7,
    "normative": 7,
    "temporal": 6,
    "negative": 5,
    "multi_hop": 5,
    "unanswerable": 3,
}

PROMPT_TYPES = {
    "factual": "Búsqueda factual: precio, cantidad, horario, dirección, actividad o condición concreta.",
    "entity": "Filtro por entidad: pregunta por una instalación, piscina, polideportivo o actividad concreta.",
    "comparison": "Comparación: contrasta dos precios, perfiles, instalaciones, modalidades o condiciones.",
    "normative": "Condición normativa: pregunta quién puede, debe, no puede o bajo qué condiciones.",
    "temporal": "Consulta temporal: usa fechas, temporada, año, periodo de apertura, vigencia o calendario.",
    "negative": "Consulta negativa: pregunta por instalaciones o servicios que no cumplen una condición.",
    "multi_hop": "Consulta multi-hop: exige combinar al menos dos hechos del contexto, como instalación + horario + precio.",
    "unanswerable": "Consulta sin respuesta: formula una pregunta plausible sobre un dato que NO aparece en el contexto proporcionado.",
}


def _text(chunk: Any) -> str:
    return str(getattr(chunk, "page_content", chunk.get("text", ""))).strip()


def _metadata(chunk: Any) -> dict[str, Any]:
    value = getattr(chunk, "metadata", None)
    if value is None and isinstance(chunk, dict):
        value = chunk.get("metadata", {})
    return dict(value or {})


def _chunk_id(chunk: Any, index: int) -> str:
    metadata = _metadata(chunk)
    for key in ("id", "chunk_id", "uid", "document_id"):
        if metadata.get(key) is not None:
            return str(metadata[key])
    if isinstance(chunk, dict):
        for key in ("id", "chunk_id", "uid"):
            if chunk.get(key) is not None:
                return str(chunk[key])
    return f"chunk_{index:08d}"


def _source(chunk: Any) -> str:
    metadata = _metadata(chunk)
    return str(metadata.get("source", metadata.get("fuente", "")))


def _json_from_llm(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S)
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise ValueError("El LLM no devolvió un objeto JSON")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("La salida del LLM no es un objeto JSON")
    return value


def _normalize(vectors: Iterable[Iterable[float]]) -> np.ndarray:
    matrix = np.asarray(list(vectors), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _nearest(query_vector: list[float], vectors: np.ndarray, ids: list[str], k: int = 10) -> list[tuple[str, float]]:
    q = _normalize([query_vector])[0]
    scores = vectors @ q
    positions = np.argsort(-scores)[:k]
    return [(ids[int(i)], float(scores[int(i)])) for i in positions]


def _balanced_sample(chunks: list[Any], n: int, rng: random.Random) -> list[tuple[int, Any]]:
    by_source: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        if _text(chunk):
            by_source[_source(chunk) or "__unknown__"].append((i, chunk))
    sources = list(by_source)
    rng.shuffle(sources)
    selected: list[tuple[int, Any]] = []
    while len(selected) < n and sources:
        next_sources = []
        for source in sources:
            if by_source[source]:
                selected.append(by_source[source].pop(rng.randrange(len(by_source[source]))))
                if len(selected) >= n:
                    break
            if by_source[source]:
                next_sources.append(source)
        sources = next_sources
    return selected


def _build_prompt(chunk_id: str, context: str, question_type: str, candidate_contexts: str) -> str:
    return f"""Eres evaluador de un sistema RAG municipal de deportes. Genera UNA pregunta de evaluación basada en el contexto.

Tipo requerido: {question_type}. {PROMPT_TYPES[question_type]}

Reglas:
- La pregunta debe estar en español y ser realista para un ciudadano.
- La respuesta debe poder justificarse únicamente con los contextos incluidos.
- No inventes precios, nombres, fechas, horarios ni permisos.
- Para multi_hop combina dos o más hechos explícitos.
- Para unanswerable pregunta por un dato plausible que no aparezca en el contexto y responde que no hay información suficiente.
- Asigna relevance por chunk: 2 = necesario/directamente responde; 1 = apoyo parcial; 0 = irrelevante.
- Usa solo IDs incluidos en los contextos.
- Devuelve SOLO JSON válido, sin Markdown.

Chunk semilla:
ID: {chunk_id}
CONTEXTO:
{context}

Contextos adicionales candidatos:
{candidate_contexts}

Formato exacto:
{{
  "query": "...",
  "reference_answer": "...",
  "relevance": {{"chunk_id": 9}},
  "question_type": "{question_type}"
}}"""


def generar_eval_set(
    chunks: list[Any],
    vectors: list[list[float]],
    embedder: Callable[[list[str]], list[list[float]]],
    llm_call: Callable[[str], str],
    n: int = 50,
    output_path: str | Path = "output/eval_queries.jsonl",
    seed: int = 42,
    type_counts: dict[str, int] | None = None,
    candidate_k: int = 8,
) -> list[dict[str, Any]]:
    """Genera y escribe registros compatibles con evaluar_rag_corpus.py.

    `embedder` debe ser exactamente el mismo modelo y configuración usados
    para los embeddings del índice.
    """
    if len(chunks) != len(vectors):
        raise ValueError("chunks y vectors deben tener la misma longitud")
    if not chunks:
        raise ValueError("No hay chunks para crear el conjunto de evaluación")

    counts = dict(type_counts or QUESTION_TYPES)
    if sum(counts.values()) != n:
        raise ValueError(f"La suma de type_counts debe ser {n}")

    rng = random.Random(seed)
    norm_vectors = _normalize(vectors)
    chunk_ids = [_chunk_id(chunk, i) for i, chunk in enumerate(chunks)]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Los chunks deben tener IDs únicos")

    seeds = _balanced_sample(chunks, n, rng)
    if len(seeds) < n:
        raise ValueError("No hay suficientes chunks no vacíos")

    type_queue = [kind for kind, count in counts.items() for _ in range(count)]
    rng.shuffle(type_queue)
    records: list[dict[str, Any]] = []

    for (seed_index, seed_chunk), question_type in zip(seeds, type_queue):
        seed_id = chunk_ids[seed_index]
        neighbors = _nearest(vectors[seed_index], norm_vectors, chunk_ids, candidate_k + 1)
        neighbor_ids = [item[0] for item in neighbors if item[0] != seed_id][:candidate_k]
        candidate_texts = []
        for neighbor_id, score in neighbors:
            if neighbor_id == seed_id:
                continue
            j = chunk_ids.index(neighbor_id)
            candidate_texts.append(f"ID: {neighbor_id} | cosine: {score:.4f}\n{_text(chunks[j])[:2500]}")
        prompt = _build_prompt(
            seed_id,
            _text(seed_chunk)[:5000],
            question_type,
            "\n\n".join(candidate_texts),
        )
        generated = _json_from_llm(llm_call(prompt))
        query = str(generated.get("query", "")).strip()
        answer = str(generated.get("reference_answer", "")).strip()
        if not query or not answer:
            raise ValueError(f"Pregunta incompleta generada para {seed_id}")

        relevance_raw = generated.get("relevance", {})
        relevance: dict[str, int] = {}
        if isinstance(relevance_raw, dict):
            for item_id, value in relevance_raw.items():
                item_id = str(item_id)
                if item_id in chunk_ids:
                    score = int(value)
                    if score in (1, 2):
                        relevance[item_id] = score
        if question_type != "unanswerable" and not relevance:
            relevance[seed_id] = 2
        if question_type == "unanswerable":
            relevance = {}

        query_vector = embedder([query])[0]
        if len(query_vector) != len(vectors[0]):
            raise ValueError("La dimensión del query_embedding no coincide con el índice")

        record = {
            "query": query,
            "query_embedding": [float(x) for x in query_vector],
            "relevant_ids": [item_id for item_id, value in relevance.items() if value > 0],
            "relevance": relevance,
            "reference_answer": answer,
            "question_type": question_type,
            "seed_chunk_id": seed_id,
            "generator_version": "index-time-v1",
        }
        records.append(record)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


if __name__ == "__main__":
    raise SystemExit("Importa generar_eval_set desde el pipeline de indexado; no se ejecuta sin tus loaders, embedder y LLM.")
