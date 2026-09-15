"""
src/index.py
Carga los embeddings en ChromaDB (base vectorial local y ligera).
"""
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from pathlib import Path
from typing import Any
from uuid import UUID

from config import CHROMA_DIR, COLLECTION_NAME, COSINE_SPACE

# Máximo de registros por llamada add(): el límite del backend varía con el
# tamaño de los vectores (chroma calcula = dim × bytes de tope). Con 2000 da margen.
_BATCH_ADD = 2_000


def obtener_guid_chroma(persist_dir: str | None = None) -> str | None:
    """Devuelve el GUID de la carpeta de datos creada por ChromaDB."""
    directorio = Path(persist_dir or CHROMA_DIR)
    if not directorio.is_dir():
        return None
    for candidato in directorio.iterdir():
        if not candidato.is_dir():
            continue
        try:
            UUID(candidato.name)
        except ValueError:
            continue
        if any(candidato.iterdir()):
            return candidato.name
    return None


def obtener_cliente_chroma(persist_dir: str | None = None) -> chromadb.ClientAPI:
    """Cliente persistente: indexas una vez, consultas muchas."""
    return chromadb.PersistentClient(
        path=persist_dir or CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def crear_coleccion(client, nombre: str | None = None):
    """Crea (o recupera) la colección. Métrica coseno, coherente con embeddings normalizados."""
    return client.get_or_create_collection(
        name=nombre or COLLECTION_NAME,
        metadata={"hnsw:space": COSINE_SPACE},
    )


# Sanitizar metadatos: Chroma no acepta None ni listas
def _sanear(meta: dict[str, Any]) -> dict[str, Any]:
    limpio = {}
    for k, v in meta.items():
        if v is None:
            limpio[k] = "null"
        elif isinstance(v, (list, tuple, set)):
            limpio[k] = str(v)
        else:
            limpio[k] = v
    return limpio


def indexar(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatos: list[dict[str, Any]],
    persist_dir: str | None = None,
    collection_name: str | None = None,
    recreate: bool = False,
) -> tuple[int, int]:
    """Inserta vectores + texto + metadata en ChromaDB.

    Returns:
        (vectores_insertados, vectores_totales_colección): la inserción
        verificada (ids vivos) y el recuento total de la colección.
    """
    client = obtener_cliente_chroma(persist_dir)
    nombre = collection_name or COLLECTION_NAME
    if recreate:
        try:
            client.delete_collection(nombre)
        except NotFoundError:
            pass  # todavía no existe
    collection = crear_coleccion(client, nombre)

    # Chroma limita los `add` por lote, con 130k en una llamada
    # estalla en "greater than max batch size (5410)". Hago un lote con margen.
    metadatos_san = [_sanear(m) for m in metadatos]
    for inicio in range(0, len(ids), _BATCH_ADD):
        fin = inicio + _BATCH_ADD
        collection.add(
            ids=ids[inicio:fin],
            embeddings=embeddings[inicio:fin],
            documents=documents[inicio:fin],
            metadatas=metadatos_san[inicio:fin],
        )
        print(f"[INDEX] insertado {min(fin, len(ids))}/{len(ids)}")

    # Verificación: todos los ids insertados en base de datos deben estar vivos.
    # `get(ids=...)` expande cada id a un parámetro SQL, lo que hace que
    # en conjunto resulte desbordada la BD: "too many SQL variables". 
    # Hago una comprobación por lotes al igual que en "batch size"
    n_unicos = len(set(ids))
    assert n_unicos == len(ids), f"IDs no únicos en la inserción ({len(ids) - n_unicos}) repetidos"
    vivos = 0
    for inicio in range(0, len(ids), _BATCH_ADD):
        fin = inicio + _BATCH_ADD
        vivos += len(collection.get(ids=ids[inicio:fin], include=[])["ids"])
    assert vivos == n_unicos, (
        f"Desajuste en nº de vectores: {vivos} vivos de {n_unicos} insertados"
    )
    totales = collection.count()
    print(f"[INDEX] colección '{collection.name}': {totales} vectores ({vivos} en esta inserción)")
    return vivos, totales