"""
src/pipeline.py
Encadena LOAD → CLEAN → TAG → CHUNK → EMBED → SCORING → DEDUP → INDEX.
Es el punto de entrada que importa tu aplicación [9].
"""
import statistics
import time

from config import TAG_SCORING_DEDUP, EXPORT_EMBEDDINGS, EMBED_DIM

from .load    import cargar_archivos
from .clean   import limpiar
from .chunk   import trocear
from .embed   import embeddear, exportar_json
from .index   import indexar
from .tsd.tag     import etiquetar
from .tsd.scoring import puntuar
from .tsd.dedup   import deduplicar


def _log(fase: str, msg: str) -> None:
    """Marca de tiempo + fase: formato único en consola (evita repetir strftime)."""
    print(time.strftime("%Y-%m-%d %H:%M:%S") + f" [{fase}] {msg}")


def _stats_chunks(chunks) -> dict:
    """Distribución de longitudes de chunk (métricas de apuntes de RAG Engineering)."""
    longitudes = [len(c.page_content) for c in chunks]
    if len(longitudes) >= 4:
        p25, p50, p75 = statistics.quantiles(longitudes, n=4)
    else:
        p25 = p50 = p75 = statistics.median(longitudes)
    return {
        "min": min(longitudes),
        "p25": p25,
        "media": round(statistics.mean(longitudes), 1),
        "p50": p50,
        "p75": p75,
        "max": max(longitudes),
        "cortos": sum(1 for l in longitudes if l < 50),  # chunks demasiado cortos (ruido)
    }


def ejecutar_pipeline(
    rutas: list[str] | str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    persist_dir: str | None = None,
    collection_name: str | None = None,
    recreate_index: bool = False,
    export_embeddings: bool = False,
) -> dict:
    """
    Ejecuta el pipeline completo de RAG (con Tagging-Scoring-Deduplication opcional).
    Returns: dict con métricas (documentos, chunks, dims, stats, dedup, tiempo).
    Uso:
        from rag_pipeline import ejecutar_pipeline
        ejecutar_pipeline("./docs/corpus/")
    """
    t0 = time.time()

    # --- LOAD
    _log("LOAD", "inicio")
    documentos = cargar_archivos(rutas)
    _log("LOAD", f"{len(documentos)} documentos cargados")

    # --- CLEAN
    _log("CLEAN", "inicio")
    documentos = limpiar(documentos)
    _log("CLEAN", f"{len(documentos)} documentos normalizados")

    # --- TAG (por fuente, antes de trocear)
    if TAG_SCORING_DEDUP:
        _log("TAG", "inicio (LLM)")
        t_tag = time.time()
        documentos = etiquetar(documentos)
        tag_metrics = {"tag_tiempo_s": round(time.time() - t_tag, 1)}

    # --- CHUNK → hereda doc_category, tags, relevancia_llm del documento padre [7]
    _log("CHUNK", "inicio")
    chunks = trocear(documentos, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    stats = _stats_chunks(chunks)
    _log("CHUNK", (f"{len(chunks)} chunks | min {stats['min']} · p25 {stats['p25']} · "
                   f"media {stats['media']} · p75 {stats['p75']} · max {stats['max']} · "
                   f"cortos(<50) {stats['cortos']}"))

    # --- EMBED
    _log("EMBED", "inicio")
    embeddings = embeddear([c.page_content for c in chunks])
    dim = len(embeddings[0]) if embeddings else 0
    _log("EMBED", f"{len(embeddings)} vectores de {dim} dims")
    
    # seguridad aplicada a dim real vs. EMBED_DIM declarado en .env (evitando el desastre)
    if dim and dim != EMBED_DIM:
        _log("EMBED", f"AVISO: dim {dim} != EMBED_DIM ({EMBED_DIM}) — "
            f"actualizad .env o regenerad el índice")

    # --- SCORING + DEDUP (requiere tag previo: relevancia_llm en metadata)
    n_pre_dedup = len(chunks)
    if TAG_SCORING_DEDUP:
        _log("TSD", "scoring + dedup")
        chunks = puntuar(chunks, embeddings)
        chunks, embeddings = deduplicar(chunks, embeddings)

    # --- INDEX
    _log("INDEX", "inicio")
    # id global estable y único: fuente + posición secuencial global
    # (chunk_index es secuencial POR DOCUMENTO, y un CSV produce muchos
    # documentos con la misma fuente, lo que haría ids duplicados y
    # Chroma sobreescribiría los chunks anteriores)
    ids = [f"{c.metadata.get('source', 'doc')}::{i}"
           for i, c in enumerate(chunks)]
    indexar(
        ids=ids,
        embeddings=embeddings,
        documents=[c.page_content for c in chunks],
        metadatos=[c.metadata for c in chunks],
        persist_dir=persist_dir,
        collection_name=collection_name,
        recreate=recreate_index,
        )

    if export_embeddings or EXPORT_EMBEDDINGS:
        exportar_json(chunks, embeddings)

    _log("FIN", f"pipeline completado en {time.time() - t0:.1f}s")
    return {
        "num_documentos": len(documentos),
        "num_chunks_pre_dedup": n_pre_dedup,
        "num_chunks_post_dedup": len(chunks),  # métrica pedida para vigilar la dedup [7]
        "chunks_descartados": n_pre_dedup - len(chunks),
        "dim_embedding": dim,
        "chunk_stats": stats,
        "tiempo_total_s": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    # Uso (desde la raíz del proyecto):
    #   python -m src.pipeline data --recreate-index
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline RAG completo: LOAD → CLEAN → TAG → CHUNK → EMBED → SCORING → DEDUP → INDEX"
    )
    parser.add_argument("rutas", nargs="+", help="Archivos o carpetas del corpus (p. ej. data)")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Longitud de chunk (por defecto: CHUNK_SIZE de .env)")
    parser.add_argument("--chunk-overlap", type=int, default=None,
                        help="Sobrelap de chunks (por defecto: CHUNK_OVERLAP de .env)")
    parser.add_argument("--persist-dir", default=None,
                        help="Directorio de ChromaDB (por defecto: CHROMA_DIR de .env)")
    parser.add_argument("--collection", default=None,
                        help="Nombre de la colección (por defecto: COLLECTION_NAME de .env)")
    parser.add_argument("--recreate-index", action="store_true",
                        help="Borrar la colección antes de indexar")
    args = parser.parse_args()

    print(ejecutar_pipeline(
        args.rutas,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        recreate_index=args.recreate_index,
    ))
    
    