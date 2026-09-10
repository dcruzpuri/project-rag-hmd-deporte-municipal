"""
src/pipeline.py
Encadena LOAD → CLEAN → CHUNK → EMBED → INDEX.
Es el punto de entrada que importa tu aplicación.
"""
import time
from config import EXPORT_EMBEDDINGS, TAG_SCORING_DEDUP, OUTPUT_DIR

from .tsd.tag import etiquetar

from .load   import cargar_archivos
from .clean  import limpiar
from .chunk  import trocear
from .embed  import embeddear, exportar_json
from .index  import indexar


def ejecutar_pipeline(
    rutas: list[str] | str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    persist_dir: str | None = None,
    collection_name: str | None = None,
    recreate_index: bool = False,
    export_embeddings: bool = EXPORT_EMBEDDINGS or False,
) -> dict:
    """
    Ejecuta el pipeline offline completo de RAG.
    devuelve dict con métricas: nº documentos, nº chunks, dimensión del vector.
    Cómo usarlo en la applicación:
        from rag_pipeline import ejecutar_pipeline
        ejecutar_pipeline("./docs/corpus/")
    """

    # 1. LOAD
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " [LOAD]  inicio de proceso...")
    documentos_crudos = cargar_archivos(rutas)
    print(time.strftime("%Y-%m-%d %H:%M:%S") + f" [LOAD]  {len(documentos_crudos)} documentos cargados")

    # 2. CLEAN
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " [CLEAN]  inicio de proceso...")
    documentos = limpiar(documentos_crudos)
    print(time.strftime("%Y-%m-%d %H:%M:%S") + f" [CLEAN] {len(documentos)} documentos normalizados")
    if TAG_SCORING_DEDUP:
        documentos = etiquetar(documentos)

    # 3. CHUNK
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " [CHUNK]  inicio de proceso...")
    chunks = trocear(documentos, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(time.strftime("%Y-%m-%d %H:%M:%S") + f" [CHUNK] {len(chunks)} chunks generados")

    # 4. EMBED
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " [EMBED]  inicio de proceso...")
    textos = [c.page_content for c in chunks]
    embeddings = embeddear(textos)
    dim = len(embeddings[0]) if embeddings else 0
    print(time.strftime("%Y.%m.%d %H:%M:%S") + f" [EMBED] {len(embeddings)} vectores de {dim} dims")
    if TAG_SCORING_DEDUP:
        from .tsd.scoring import puntuar
        from .tsd.dedup import deduplicar
        print(time.strftime("%Y-%m-%d %H:%M:%S") + " [SCORE]  inicio de proceso...")
        chunks = puntuar(chunks, embeddings)
        print(time.strftime("%Y-%m-%d %H:%M:%S") + f" [SCORE]  puntuación semántica calculada para {len(chunks)} chunks")
        print(time.strftime("%Y-%m-%d %H:%M:%S") + " [DEDUP]  inicio de proceso...")
        chunks, embeddings = deduplicar(chunks, embeddings)
        print(time.strftime("%Y-%m-%d %H:%M:%S") + f" [DEDUP]  {len(chunks)} chunks tras deduplicación")
    
    # Opcional: exportar para inspección
    if export_embeddings:
        datos = [
            {"text": c.page_content, "metadata": c.metadata, "embedding": e}
            for c, e in zip(chunks, embeddings)
        ]
        exportar_json(datos)
        print(time.strftime("%Y-%m-%d %H:%M:%S") + f" [EMBED] Embeddings exportados a {OUTPUT_DIR}/embeddings.json")

    # 5. INDEX
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " [INDEX]  inicio de proceso...")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    indexar(
        ids=ids,
        embeddings=embeddings,
        documents=textos,
        metadatas=[c.metadata for c in chunks],
        persist_dir=persist_dir,
        collection_name=collection_name,
        recreate=recreate_index,
    )
    print(time.strftime("%Y-%m-%d %H:%M:%S") + f" [INDEX] {len(ids)} vectores indexados en ChromaDB")

    return {
        "num_documents": len(documentos_crudos),
        "num_chunks": len(chunks),
        "embedding_dim": dim,
    }