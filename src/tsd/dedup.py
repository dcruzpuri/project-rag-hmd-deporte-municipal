"""
src/tsd/dedup.py
Deduplicación consciente de la política de csv_advisor: los chunks cuya
metadata lleva `dedup_policy` exacta (exact_key / group_only / exact_only) se
dedup por clave y NUNCA por coseno (evita perder IDs o grupos casi iguales).
Los demás (sin política, o `semantic_optional`) pasan por la deduplicación
semántica: similitud coseno umbral + `semantic_score` del scoring.

--- ATENCIÓN ---
◬ Los chunks semánticos están vinculados al scoring de donde toman la mayor
  puntuación semántica en la comparación de chunks (requiere scoring previo).
--- INFORMACIÓN DE CONFIGURACIÓN ---
        → Umbral configurable en archivo .env: DEDUP_UMBRAL

Implementación incremental con FAISS (IndexFlatIP): cada candidato se busca
contra el índice plano que va creciendo con los chunks ya conservados.
Semántica idéntica a la matriz completa `sim[i, j] < umbral`, pero la memoria
es O(n·d) en vez de O(n²) (~1,3 GB en vez de ~68 GB para n=129.917, d=2.560).
"""

import time
from collections import Counter
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document

from config import DEDUP_UMBRAL
from src.tsd.tag import CATEGORIAS_VALIDAS

_POLICIAS_EXACTAS = frozenset({"exact_only", "exact_key", "group_only"})


def _cobertura_categoria(
    chunks_pre: list[Document], chunks_post: list[Document]
) -> dict[str, dict[str, int]]:
    """Chunks por doc_category, fija a la taxonomía cerrada (las 6 siempre, pre/post).

    Cubre el hueco del informe: una categoría a 0 post-dedup avisa de que la
    intención primaria de esa fuente ya no responde a preguntas del dominio.
    """
    cov = {cat: {"pre": 0, "post": 0} for cat in sorted(CATEGORIAS_VALIDAS)}
    for chunk in chunks_pre:
        cat = chunk.metadata.get("doc_category")
        if cat in cov:
            cov[cat]["pre"] += 1
    for chunk in chunks_post:
        cat = chunk.metadata.get("doc_category")
        if cat in cov:
            cov[cat]["post"] += 1
    return cov


def _tags_top3(chunks_post: list[Document]) -> list[tuple[str, int]]:
    """Top 3 tags (booleanos `tag_<nombre>`) por nº de chunks conservados.

    Empates por nombre alfabético (orden determinista). La multi-facialidad ya
    vive en la metadata; aquí solo se resume para la lectura del informe.
    """
    conteo: dict[str, int] = {}
    for chunk in chunks_post:
        for clave, val in chunk.metadata.items():
            if clave.startswith("tag_") and val is True:
                tag = clave[len("tag_"):]
                conteo[tag] = conteo.get(tag, 0) + 1
    return sorted(conteo.items(), key=lambda kv: (-kv[1], kv[0]))[:3]


def _clave_exacta(chunk: Document) -> tuple:
    """Clave de identidad para las políticas exactas (fuente + clave de la entidad/grupo/fila)."""
    policy = chunk.metadata.get("dedup_policy")
    if policy == "exact_key":
        return (
            chunk.metadata.get("source"),
            "entity",
            chunk.metadata.get("entity_key"),
        )
    if policy == "group_only":
        return (
            chunk.metadata.get("source"),
            "group",
            chunk.metadata.get("group_key"),
            chunk.metadata.get("chunk_index"),
        )
    # exact_only (y el resto no semántico): contenido literal de la fila
    return (
        chunk.metadata.get("source"),
        "row",
        chunk.metadata.get("row"),
        chunk.page_content,
    )


def _deduplicar_semantico(
    chunks: list[Document], embeddings: list[list[float]], umbral: float
) -> tuple[list[int], list[list[float]], list[tuple[int, float, int]]]:
    """Descarta chunks cercanos a uno ya conservado (score descendente).

    Returns:
        (índices conservados EN ORDEN DE ENTRADA, embeddings correspondientes,
        eventos de descarte [(i, similitud, i_ganador), ...] donde los índices son
        posiciones sublista y ``i_ganador`` la posición del chunk que venció).
    """
    if not chunks:
        return [], [], []

    E = np.asarray(embeddings, dtype=np.float32)
    dim = E.shape[1]

    orden = sorted(
        range(len(chunks)),
        key=lambda i: chunks[i].metadata.get("semantic_score", 0.0),
        reverse=True,
    )

    kept: list[int] = []
    eventos: list[tuple[int, float, int]] = []
    idx = faiss.IndexFlatIP(dim)  # fuerza bruta exacta, sin entrenamiento
    for i in orden:
        # ¿Existe un chunk conservado con similitud >= umbral? si sí, se descarta.
        # Con el índice vacío, el primer chunk entra directamente.
        if idx.ntotal:
            D, I = idx.search(E[i : i + 1], 1)
            if D[0][0] >= umbral:
                # I[0][0] = sitio del ganador dentro de `idx` = kept[j] (orden añadido)
                eventos.append((i, float(D[0][0]), kept[I[0][0]]))
                continue
        idx.add(E[i : i + 1])
        kept.append(i)

    kept.sort()  # conservar el orden original en la salida
    return kept, [embeddings[i] for i in kept], eventos


def deduplicar(
    chunks: list[Document], embeddings: list[list[float]], umbral: float = DEDUP_UMBRAL,
    info: dict[str, Any] | None = None,
) -> tuple[list[Document], list[list[float]]]:
    """Dedup por política (csv_advisor) + dedup semántica del resto.

    Los chunks con `dedup_policy` exacta se dedup por clave (solo se descarta
    la repetición literal) y el resto pasa por la deduplicación semántica
    (score descendente + FAISS), que es la que requiere `semantic_score`.

    Args:
        umbral: umbral coseno de similitud para descartar duplicados.
        info: dict opcional donde se vuelcan las métricas del dedup
            (para el informe de indexación), incluyendo
            «chunks_por_categoria» (las 6 de la taxonomía cerrada, pre/post),
            «tags_top3_post» (resumen compacto de la multi-facialidad de los
            chunks conservados) y la auditoría de pares descartados:
            «descartes» (uno por chunk: motivo dedup_coseno/dedup_clave,
            similitud del par, chunk afectado y chunk vencedor con sus
            fuentes y snippets), «descartes_por_fuente» (n por fuente) y
            «fuentes_vacias» (fuentes que entraron y salieron a 0, con el
            descarte que las vació).
    """
    t0 = time.perf_counter()
    if not chunks:
        return [], []

    semantic: list[int] = []
    passthrough: list[int] = []
    n_descartadas_exatas = 0
    vistos: set[tuple] = set()
    eventos: list[dict[str, Any]] = []  # auditoría: motivo, chunk afectado, pareja

    def _ident(chunk: Document) -> dict[str, Any]:
        m = chunk.metadata
        return {
            "fuente": m.get("source", "?"),
            "chunk_index": m.get("chunk_index"),
            "snip": (str(chunk.page_content) or "").strip().replace("\n", " ")[:120],
            "score": m.get("semantic_score", 0.0),
        }

    for i, chunk in enumerate(chunks):
        policy = chunk.metadata.get("dedup_policy")
        if policy not in _POLICIAS_EXACTAS:
            semantic.append(i)
            continue
        clave = _clave_exacta(chunk)
        if clave in vistos:
            n_descartadas_exatas += 1
            eventos.append({
                "motivo": "dedup_clave",
                "policy": policy,
                "fuente": chunk.metadata.get("source", "?"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "snip": (str(chunk.page_content) or "").strip().replace("\n", " ")[:120],
            })
            continue
        vistos.add(clave)
        passthrough.append(i)

    kept: set[int] = set(passthrough)
    if semantic:
        sem_chunks = [chunks[i] for i in semantic]
        sem_embs = [embeddings[i] for i in semantic]
        sem_kept_pos, _, desc_coseno = _deduplicar_semantico(sem_chunks, sem_embs, umbral)
        kept.update(semantic[j] for j in sem_kept_pos)
        for i_sem, sim, j_gan in desc_coseno:
            vict, derro = sem_chunks[j_gan], sem_chunks[i_sem]
            eventos.append({
                "motivo": "dedup_coseno",
                "sim": round(float(sim), 4),
                "fuente": derro.metadata.get("source", "?"),
                "chunk_index": derro.metadata.get("chunk_index"),
                "policy": derro.metadata.get("dedup_policy"),
                "snip": (str(derro.page_content) or "").strip().replace("\n", " ")[:120],
                "pareja": {
                    "fuente": vict.metadata.get("source", "?"),
                    "chunk_index": vict.metadata.get("chunk_index"),
                    "score": vict.metadata.get("semantic_score", 0.0),
                    "snip": (str(vict.page_content) or "").strip().replace("\n", " ")[:120],
                },
            })

    # Salir en orden original (passthrough + semánticos intercalados).
    kept_orden = sorted(kept)

    n_total = len(chunks)
    n_descartadas = n_total - len(kept_orden)
    post_chunks = [chunks[i] for i in kept_orden]
    n_desc_sem = n_descartadas - n_descartadas_exatas
    print(
        f"[DEDUP] umbral {umbral} -> descarta {n_descartadas} de {n_total} "
        f"({n_descartadas / n_total:.1%}) | passthrough (política exacta) "
        f"{len(passthrough)} descartados exactos {n_descartadas_exatas} · semánticos {len(semantic)}"
    )
    if info is not None:
        por_fuente_n = Counter(ev["fuente"] for ev in eventos)
        # Fuentes que entraron (pre > 0) y salieron vacías (post == 0): la evidencia
        # es el descarte coseno de mayor similitud o, si vació por clave, el último.
        pre_post = {
            c.metadata.get("source", "?")
            for c in chunks
        } - {c.metadata.get("source", "?") for c in post_chunks}
        vacias: dict[str, dict[str, Any]] = {}
        for ev in eventos:
            f = ev["fuente"]
            if f in pre_post:
                prev = vacias.get(f)
                sim_nueva = ev.get("sim", float("-inf"))
                sim_prev = (prev or {}).get("sim", float("-inf"))
                if prev is None or sim_nueva > sim_prev:
                    vacias[f] = ev
        info["dedup"] = {
            "umbral": umbral,
            "chunks_pre": n_total,
            "chunks_post": len(kept_orden),
            "descartados_exactos": n_descartadas_exatas,
            "descartados_semantico": n_desc_sem,
            "descartados_total": n_descartadas,
            "descartados_pct": round(n_descartadas / n_total * 100, 1),
            "tiempo_s": round(time.perf_counter() - t0, 3),
            "chunks_por_categoria": _cobertura_categoria(chunks, post_chunks),
            "tags_top3_post": _tags_top3(post_chunks),
            "descartes": eventos,
            "descartes_por_fuente": dict(sorted(por_fuente_n.items())),
            "fuentes_vacias": dict(sorted(vacias.items())),
        }
    return post_chunks, [embeddings[i] for i in kept_orden]
