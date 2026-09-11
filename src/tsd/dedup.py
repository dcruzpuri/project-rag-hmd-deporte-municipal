"""
src/tsd/dedup.py
Deduplicación por similitud coseno. Descarta chunks con embeddings 
similares y conserva los de mayor puntuación semántica.
--- ATENCIÓN ---
◬ Está vinculado al scoring de donde toma la moyor puntuación 
   semántica en la comparación de chunks.
--- INFORMACIÓN DE CONFIGURACIÓN ---
        → Umbral configurable en archivo .env: DEDUP_UMBRAL
"""
import numpy as np
from langchain_core.documents import Document

from config import DEDUP_UMBRAL


def deduplicar(chunks: list[Document], embeddings: list[list[float]],
               umbral: float = DEDUP_UMBRAL) -> tuple[list[Document], list[list[float]]]:
    # Recorre por semantic_score descendente y descarta los casi idénticos
    E = np.asarray(embeddings, dtype=np.float32)
    sim = E @ E.T
    # Se prioriza el chunk mejor puntuado: si varios son casi iguales, se queda el mejor
    orden = sorted(range(len(chunks)),
                   key=lambda i: chunks[i].metadata.get("semantic_score", 0.0),
                   reverse=True)
    kept: list[int] = []
    for i in orden:
        if all(sim[i, j] < umbral for j in kept):
            kept.append(i)
    kept.sort()  # conservar el orden original en la salida

    if not chunks:
        return [], []
    n_descartados = len(chunks) - len(kept)
    print(f"[DEDUP] umbral {umbral} -> descarta {n_descartados} de {len(chunks)} "
          f"({n_descartados / len(chunks):.1%})")
    return [chunks[i] for i in kept], [embeddings[i] for i in kept]