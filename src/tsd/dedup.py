"""
src/tsd/dedup.py
Deduplicación semántica greedy por similitud coseno.
Descarta chunks con embeddings muy similares (umbral configurable) 
y conserva los de mayor puntuación semántica.
"""
import numpy as np
from langchain_core.documents import Document

def deduplicar(chunks: list[Document], embeddings: list[list[float]],
               umbral: float = 0.93) -> tuple[list[Document], list[list[float]]]:
    E = np.asarray(embeddings, dtype=np.float32)
    sim = E @ E.T
    orden = sorted(range(len(chunks)),
                   key=lambda i: chunks[i].metadata.get("semantic_score", 0.0),
                   reverse=True)
    kept: list[int] = []
    for i in orden:
        if all(sim[i, j] < umbral for j in kept):
            kept.append(i)
    kept.sort()
    return [chunks[i] for i in kept], [embeddings[i] for i in kept]