"""
src/tsd/scoring.py
semantic_score = 0.45·relevancia_LLM + 0.20·centralidad
                 + 0.20·(1 − redundancia) + 0.15·autoridad  [2]
Combina la relevancia del LLM con métricas reales de los embeddings
(centralidad y redundancia) y la autoridad de la fuente.

La redundancia se obtiene con FAISS (búsqueda exacta sobre índice plano)
en vez de materializar la matriz de similitud n×n: la memoria pasa de
O(n²) a O(n·d) (~68 GB a ~1,3 GB en la escala real: n=129.917, d=2.560),
que era la causa del OOM en la fase [TSD].
"""

import time
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document

from config import DEDUP_UMBRAL

AUTORIDAD = {
    "reglamento": 1.0,
    "normativa": 1.0,
    "precios": 0.9,
    "tarifas": 0.9,
    "agenda": 0.6,
}


def _autoridad(source: str) -> float:
    """Peso por tipo de fuente: reglamentos/normativas pesan más que un CSV genérico."""
    s = source.lower()
    for clave, peso in AUTORIDAD.items():
        if clave in s:
            return peso
    return 0.7  # CSV genérico


def _redundancias(E: np.ndarray) -> np.ndarray:
    """Similitud coseno de cada vector con su vecino más cercano (sin contar el propio).

    Args:
        E: Matriz (n, d) de vectores float32 (normalizados por el proveedor).

    Returns:
        Vector (n,) con la similitud máxima de cada fila contra el resto
        (0.0 si solo existe un vector o si todas las similitudes son negativas).
    """
    n, dim = E.shape
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    idx = faiss.IndexFlatIP(dim)
    idx.add(E)
    D, I = idx.search(E, 2)
    # k=2 cubre el máximo: la mejor similitud de una fila siempre está en el
    # top-2 (el propio + su mejor vecino, en los dos órdenes posibles).
    es_propio = I == np.arange(n, dtype=np.int64)[:, None]
    D[es_propio] = 0.0  # misma semántica que el antiguo fill_diagonal(sim, 0.0)
    red: np.ndarray = D.max(axis=1)
    return red.astype(np.float32)


def _redundancia_por_politica(
    chunks: list[Document], redundancias: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Redundancia (similitud máx vecino más cercano) agrupada por política de dedup.

    Los chunks sin ``dedup_policy`` (o ``semantic_optional``) se agrupan como
    ``semantica``: son los que de verdad pasan por la dedup coseno.
    Returns:
        {política: {n, media, p50, p90, pct_sup_umbral}} (p50/p90 con numpy para
        soportar n=1; los valores se redondean a 4 decimales para el informe).
    """
    grupos: dict[str, list[float]] = {}
    for i, chunk in enumerate(chunks):
        pol = chunk.metadata.get("dedup_policy") or "semantica"
        grupos.setdefault(pol, []).append(float(redundancias[i]))
    por_pol: dict[str, dict[str, Any]] = {}
    for pol, vals in grupos.items():
        a = np.asarray(vals, dtype=np.float32)
        por_pol[pol] = {
            "n": int(a.size),
            "media": round(float(a.mean()), 4),
            "p50": round(float(np.median(a)), 4),
            "p90": round(float(np.percentile(a, 90)), 4),
            "pct_sup_umbral": round(float((a >= DEDUP_UMBRAL).mean()) * 100, 1),
        }
    return por_pol


def puntuar(chunks: list[Document], embeddings: list[list[float]],
            info: dict[str, Any] | None = None) -> list[Document]:
    """Puntúa cada chunk y guarda el resultado en metadata['semantic_score'].

    Args:
        chunks: chunks con metadata (relevancia_llm, source).
        embeddings: matrices de los vectores (float32, normalizados).
        info: dict opcional donde se vuelcan las métricas del scoring
            (para el informe de indexación).
    """
    if not chunks:
        return chunks

    # Mide la fase COMPLETA (array + centroide + redundancia FAISS + bucle):
    # perf_counter para precisión real (los tiempos del informe muestran ms).
    t0 = time.perf_counter()
    E = np.asarray(embeddings, dtype=np.float32)
    n, dim = E.shape

    # Centralidad: coseno del chunk contra el centroide (media de la colección).
    centroide = E.mean(axis=0) if n else np.zeros(dim, dtype=np.float32)
    norm_centroide = np.linalg.norm(centroide)
    if norm_centroide > 0:
        centroide /= norm_centroide
    centralidad = E @ centroide

    # Redundancia: FAISS fuerza bruta (memoria O(n·d), tiempo O(n²·d) en C++
    # multithread) en lugar de E @ E.T completo. Igual que el original, asume
    # vectores normalizados y trata el producto punto como coseno.
    redundancias = _redundancias(E)

    for i, chunk in enumerate(chunks):
        m = chunk.metadata
        score = (
            0.45 * float(m.get("relevancia_llm", 0.0))
            + 0.20 * float(centralidad[i])
            + 0.20 * (1.0 - float(redundancias[i]))
            + 0.15 * _autoridad(m.get("source", ""))
        )
        chunk.metadata["semantic_score"] = round(min(max(score, 0.0), 1.0), 4)

    # Resumen por consola: mín / media / máx + mejor chunk
    scores = [c.metadata["semantic_score"] for c in chunks]
    top = max(chunks, key=lambda c: c.metadata["semantic_score"])
    print(
        f"[SCORE] {len(chunks)} chunks | mín {min(scores):.3f} · "
        f"media {sum(scores) / len(scores):.3f} · máx {max(scores):.3f}"
    )
    print(
        f"[SCORE] mejor: \"{top.page_content[:60].strip()}...\" ({top.metadata['semantic_score']:.3f})"
    )
    if info is not None:
        info.update(
            {
                "scoring": {
                    "semantic_score_min": round(min(scores), 4),
                    "semantic_score_media": round(sum(scores) / len(scores), 4),
                    "semantic_score_max": round(max(scores), 4),
                    "centralidad_media": round(float(centralidad.mean()), 4),
                    "redundancia_media": round(float(redundancias.mean()), 4),
                    "score_buenos_n": sum(1 for s in scores if s >= 0.6),
                    "chunks_n": len(scores),
                    "score_buenos_pct": round(
                        sum(1 for s in scores if s >= 0.6) / len(scores) * 100, 2
                    ),
                    "tiempo_s": round(time.perf_counter() - t0, 3),
                    # la redundancia media estructural (group_only/exact_key alta)
                    # solo es legible segmentada por política: el umbral y el
                    # % ≥ umbral se calculan con DEDUP_UMBRAL (el que decide el dedup)
                    "redundancia_umbral": DEDUP_UMBRAL,
                    "redundancia_por_politica": _redundancia_por_politica(
                        chunks, redundancias
                    ),
                }
            }
        )
    return chunks
