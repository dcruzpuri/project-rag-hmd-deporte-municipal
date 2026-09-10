"""
src/index.py
Carga los embeddings en ChromaDB (base vectorial local y ligera).
"""

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, COLLECTION_NAME, COSINE_SPACE


def obtener_cliente_chroma(persist_dir: str | None = None) -> chromadb.ClientAPI:
    """Cliente persistente: indexas una vez, consultas muchas."""
    return chromadb.PersistentClient(
        path=persist_dir or CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def crear_coleccion(client, nombre: str | None = None):
    """
    Crea (o recupera) la colección.
    Métrica coseno, coherente con embeddings normalizados.
    """
    return client.get_or_create_collection(
        name=nombre or COLLECTION_NAME,
        metadata={"hnsw:space": COSINE_SPACE},
    )


def indexar(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
    persist_dir: str | None = None,
    collection_name: str | None = None,
    recreate: bool = False,
) -> None:
    """
    Inserta vectores + texto + metadata en ChromaDB.

    Reglas:
      - ids únicos.
      - embeddings / documents / metadatas alineados por posición con ids.
      - metadatas solo str, int, float, bool (convertir None → "null").
    """
    client = obtener_cliente_chroma(persist_dir)
    nombre = collection_name or COLLECTION_NAME

    if recreate:
        try:
            client.delete_collection(nombre)
        except Exception:
            pass

    collection = crear_coleccion(client, nombre)

    # Sanitizar metadatos: Chroma no acepta None ni listas
    def _sanear(meta: dict) -> dict:
        limpio = {}
        for k, v in meta.items():
            if v is None:
                limpio[k] = "null"
            elif isinstance(v, (list, tuple, set)):
                limpio[k] = str(v)
            else:
                limpio[k] = v
        return limpio

    metadatas_san = [_sanear(m) for m in metadatas]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas_san,
    )

    # Verificación rápida
    assert collection.count() == len(ids), "Desajuste en nº de vectores"