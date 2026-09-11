"""
src/tsd/scoring.py
semantic_score = 0.45·relevancia_LLM + 0.20·centralidad
                 + 0.20·(1 − redundancia) + 0.15·autoridad  [2]
Combina la relevancia del LLM con métricas reales de los embeddings
(centralidad y redundancia) y la autoridad de la fuente.
"""
import numpy as np
from langchain_core.documents import Document

AUTORIDAD = {"reglamento": 1.0, "normativa": 1.0,
             "precios": 0.9, "tarifas": 0.9,
             "agenda": 0.6}


def _autoridad(source: str) -> float:
    """Peso por tipo de fuente: reglamentos/normativas pesan más que un CSV genérico."""
    s = source.lower()
    for clave, peso in AUTORIDAD.items():
        if clave in s:
            return peso
    return 0.7  # CSV genérico


def puntuar(chunks: list[Document], embeddings: list[list[float]]) -> list[Document]:
    """Puntúa cada chunk y guarda el resultado en metadata['semantic_score']."""
    E = np.asarray(embeddings, dtype=np.float32)
    centroide = E.mean(axis=0)
    norm_centroide = np.linalg.norm(centroide)
    if norm_centroide > 0:
        centroide /= norm_centroide
    sim = E @ E.T  # coseno (vectores normalizados)
    # BUG corregido: la diagonal era 1.0 (auto-similitud), así que redundancia
    # salía siempre 1.0 y el término 0.20·(1−redundancia) era siempre 0.
    np.fill_diagonal(sim, 0.0)

    for i, chunk in enumerate(chunks):
        m = chunk.metadata
        centralidad = float(E[i] @ centroide)
        redundancia = float(sim[i].max()) if len(chunks) > 1 else 0.0
        score = (0.45 * float(m.get("relevancia_llm", 0.0))
                 + 0.20 * centralidad
                 + 0.20 * (1.0 - redundancia)
                 + 0.15 * _autoridad(m.get("source", "")))
        chunk.metadata["semantic_score"] = round(min(max(score, 0.0), 1.0), 4)

    # Resumen por consola: mín / media / máx + mejor chunk
    scores = [c.metadata["semantic_score"] for c in chunks]
    top = max(chunks, key=lambda c: c.metadata["semantic_score"])
    print(f"[SCORE] {len(chunks)} chunks | mín {min(scores):.3f} · "
          f"media {sum(scores) / len(scores):.3f} · máx {max(scores):.3f}")
    print(f"[SCORE] mejor: \"{top.page_content[:60].strip()}...\" ({top.metadata['semantic_score']:.3f})")
    return chunks