"""
scripts/benchmark_tsd.py
Verifica a escala real el coste de la fase TSD (scoring + dedup) tras
la migración a FAISS. Carga output/embeddings.json (el export propio del
pipeline con EXPORT_EMBEDDINGS=true) y ejecuta scoring + dedup sin tocar
ChromaDB ni Ollama.

Ejecutar desde la raíz del proyecto:
    python -m scripts.benchmark_tsd
"""

import json
import time

from langchain_core.documents import Document

from src.tsd.scoring import puntuar
from src.tsd.dedup import deduplicar


def _cargar(ruta: str) -> tuple[list[Document], list[list[float]]]:
    """Carga el dump JSON del pipeline (text + metadata + embedding)."""
    with open(ruta, encoding="utf-8") as f:
        registros = json.load(f)
    docs = [
        Document(page_content=r["text"], metadata=dict(r["metadata"]))
        for r in registros
    ]
    vecs = [list(r["embedding"]) for r in registros]
    return docs, vecs


def main() -> None:
    docs, vecs = _cargar("output/embeddings.json")
    n, d = len(docs), len(vecs[0])
    print(
        f"[BENCH] corpus real: n={n}, d={d} ({n * d * 4 / 1e9:.3f} GB de vectores f32)"
    )

    t0 = time.time()
    puntuar(docs, vecs)  # escribe metadata['semantic_score'] por referencia
    t_score = time.time() - t0
    print(f"[BENCH] puntuar (scoring FAISS k=2) en {t_score:.1f}s")

    t0 = time.time()
    chunks_kept, vecs_kept = deduplicar(docs, vecs)
    t_dedup = time.time() - t0
    print(
        f"[BENCH] deduplicar (greedy incremental) en {t_dedup:.1f}s "
        f"-> conserva {len(chunks_kept)} de {n}"
    )

    print(
        f"[BENCH] TSD total: {t_score + t_dedup:.1f}s "
        f"(la matriz E@E.T habría sido {n * n * 4 / 1e9:.0f} GB -> OOM)"
    )


if __name__ == "__main__":
    main()
