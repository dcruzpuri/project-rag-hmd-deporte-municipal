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
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document

from config import DEDUP_UMBRAL

_POLICIAS_EXACTAS = frozenset({"exact_only", "exact_key", "group_only"})


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
) -> tuple[list[int], list[list[float]]]:
    """Descarta chunks cercanos a uno ya conservado (score descendente).

    Returns:
        (índices conservados EN ORDEN DE ENTRADA, embeddings correspondientes).
    """
    if not chunks:
        return [], []

    E = np.asarray(embeddings, dtype=np.float32)
    dim = E.shape[1]

    orden = sorted(
        range(len(chunks)),
        key=lambda i: chunks[i].metadata.get("semantic_score", 0.0),
        reverse=True,
    )

    kept: list[int] = []
    idx = faiss.IndexFlatIP(dim)  # fuerza bruta exacta, sin entrenamiento
    for i in orden:
        # ¿Existe un chunk conservado con similitud >= umbral? si sí, se descarta.
        # Con el índice vacío, el primer chunk entra directamente.
        if idx.ntotal and idx.search(E[i : i + 1], 1)[0][0][0] >= umbral:
            continue
        idx.add(E[i : i + 1])
        kept.append(i)

    kept.sort()  # conservar el orden original en la salida
    return kept, [embeddings[i] for i in kept]


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
            (para el informe de indexación).
    """
    t0 = time.time()
    if not chunks:
        return [], []

    semantic: list[int] = []
    passthrough: list[int] = []
    n_descartadas_exatas = 0
    vistos: set[tuple] = set()

    for i, chunk in enumerate(chunks):
        policy = chunk.metadata.get("dedup_policy")
        if policy not in _POLICIAS_EXACTAS:
            semantic.append(i)
            continue
        clave = _clave_exacta(chunk)
        if clave in vistos:
            n_descartadas_exatas += 1
            continue
        vistos.add(clave)
        passthrough.append(i)

    kept: set[int] = set(passthrough)
    if semantic:
        sem_chunks = [chunks[i] for i in semantic]
        sem_embs = [embeddings[i] for i in semantic]
        sem_kept_pos, _ = _deduplicar_semantico(sem_chunks, sem_embs, umbral)
        kept.update(semantic[j] for j in sem_kept_pos)

    # Salir en orden original (passthrough + semánticos intercalados).
    kept_orden = sorted(kept)

    n_total = len(chunks)
    n_descartadas = n_total - len(kept_orden)
    print(
        f"[DEDUP] umbral {umbral} -> descarta {n_descartadas} de {n_total} "
        f"({n_descartadas / n_total:.1%}) | passthrough (política exacta) "
        f"{len(passthrough)} descartados exactos {n_descartadas_exatas} · semánticos {len(semantic)}"
    )
    if info is not None:
        n_desc_sem = n_descartadas - n_descartadas_exatas
        info["dedup"] = {
            "umbral": umbral,
            "chunks_pre": n_total,
            "chunks_post": len(kept_orden),
            "descartados_exactos": n_descartadas_exatas,
            "descartados_semantico": n_desc_sem,
            "descartados_total": n_descartadas,
            "descartados_pct": round(n_descartadas / n_total * 100, 1),
            "tiempo_s": round(time.time() - t0, 1),
        }
    return [chunks[i] for i in kept_orden], [embeddings[i] for i in kept_orden]
