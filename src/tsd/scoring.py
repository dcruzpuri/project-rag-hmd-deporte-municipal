"""
src/tsd/scoring.py
semantic_score = 0.45·relevancia_LLM + 0.20·centralidad
               + 0.20·(1 − redundancia) + 0.15·autoridad
Scoring de contenidos para RAG (retrieval-augmented generation) con LLM local (Ollama). 
Combina la relevancia del LLM con metricas reales de los embeddings
(centralidad y redundancia) y la autoridad de la fuente.
"""
import numpy as np
from langchain_core.documents import Document

AUTORIDAD = {"reglamento": 1.0, "normativa": 1.0,
             "precios": 0.9, "tarifas": 0.9,
             "agenda": 0.6}

def _autoridad(source: str) -> float:
    s = source.lower()
    for clave, peso in AUTORIDAD.items():
        if clave in s:
            return peso
    return 0.7  # CSV genérico

def puntuar(chunks: list[Document], embeddings: list[list[float]]) -> list[Document]:
    E = np.asarray(embeddings, dtype=np.float32)
    centroide = E.mean(axis=0)
    centroide /= np.linalg.norm(centroide)
    sim = E @ E.T  # coseno (vectores normalizados)
    for i, chunk in enumerate(chunks):
        m = chunk.metadata
        centralidad = float(E[i] @ centroide)
        redundancia = float(sim[i].max())
        score = (0.45 * float(m.get("relevancia_llm", 0.0))
                 + 0.20 * centralidad
                 + 0.20 * (1.0 - redundancia)
                 + 0.15 * _autoridad(m.get("source", "")))
        chunk.metadata["semantic_score"] = round(min(max(score, 0.0), 1.0), 4)
    return chunks