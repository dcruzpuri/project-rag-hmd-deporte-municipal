"""
src/tsd/dedup.py
Deduplicación por similitud coseno. Descarta chunks con embeddings
similares y conserva los de mayor puntuación semántica.
--- ATENCIÓN ---
◬ Está vinculado al scoring de donde toma la mayor puntuación
    semántica en la comparación de chunks.
--- INFORMACIÓN DE CONFIGURACIÓN ---
        → Umbral configurable en archivo .env: DEDUP_UMBRAL

Implementación incremental con FAISS: se recorre por `semantic_score`
descendente y cada chunk candidato se busca contra el índice plano que
va creciendo con los chunks ya conservados. Semántica idéntica a la
matriz completa `sim[i, j] < umbral`, pero la memoria es O(n·d) en vez
de O(n²) (~1,3 GB en vez de ~68 GB para n=129.917, d=2.560).
"""

import faiss
import numpy as np
from langchain_core.documents import Document

from config import DEDUP_UMBRAL


def deduplicar(
    chunks: list[Document], embeddings: list[list[float]], umbral: float = DEDUP_UMBRAL
) -> tuple[list[Document], list[list[float]]]:
    """Descarta chunks cercanos a uno ya conservado (score descendente).

    Args:
        chunks: chunks con `semantic_score` en metadata (requiere scoring previo).
        embeddings: vectores float correspondientes a `chunks`, mismo orden.
        umbral: similitud coseno mínima para considerar dos chunks duplicados.

    Returns:
        (chunks conservados, embeddings correspondientes), en orden original.
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

    n_descartados = len(chunks) - len(kept)
    print(
        f"[DEDUP] umbral {umbral} -> descarta {n_descartados} de {len(chunks)} "
        f"({n_descartados / len(chunks):.1%})"
    )
    return [chunks[i] for i in kept], [embeddings[i] for i in kept]
