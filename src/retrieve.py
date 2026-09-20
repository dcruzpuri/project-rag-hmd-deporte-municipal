"""
src/retrieve.py

Recuperación semántica desde ChromaDB.

Responsabilidad:
    pregunta -> embedding de consulta -> búsqueda vectorial -> top-k resultados

Este módulo NO genera respuestas con un LLM.
Solo recupera contexto relevante para que generate.py / logic.py
puedan utilizarlo posteriormente.
"""

from typing import Any

from config import CHROMA_DIR, COLLECTION_NAME
from src.embed import embeddear_consulta
from src.index import obtener_cliente_chroma, crear_coleccion


def obtener_coleccion(
    persist_dir: str | None = None,
    collection_name: str | None = None,
):
    """
    Abre la colección persistente de ChromaDB utilizada por el índice.

    Args:
        persist_dir: directorio de Chroma. Si no se indica, usa CHROMA_DIR.
        collection_name: nombre de la colección. Si no se indica,
                         usa COLLECTION_NAME.

    Returns:
        Colección ChromaDB.
    """
    client = obtener_cliente_chroma(persist_dir or CHROMA_DIR)
    return crear_coleccion(
        client,
        collection_name or COLLECTION_NAME,
    )


def recuperar(
    pregunta: str,
    top_k: int = 5,
    persist_dir: str | None = None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Recupera los chunks más similares a una pregunta.

    Flujo:
        1. Genera embedding de la pregunta.
        2. Consulta ChromaDB mediante similitud vectorial.
        3. Devuelve los top-k resultados.

    Args:
        pregunta: consulta del usuario.
        top_k: número máximo de chunks a recuperar.
        persist_dir: directorio de ChromaDB.
        collection_name: colección que se consulta.

    Returns:
        Lista de diccionarios con:
            - text: contenido del chunk
            - source: fuente del chunk
            - metadata: metadata completa
            - distance: distancia devuelta por Chroma

    Raises:
        ValueError: si la pregunta está vacía o top_k no es válido.
    """

    if not isinstance(pregunta, str) or not pregunta.strip():
        raise ValueError("La pregunta no puede estar vacía.")

    if top_k <= 0:
        raise ValueError("top_k debe ser mayor que 0.")

    # Embedding de consulta usando exactamente el mismo proveedor/modelo
    # configurado para los embeddings del índice.
    embedding = embeddear_consulta(pregunta.strip())

    collection = obtener_coleccion(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    if collection.count() == 0:
        return []

    # No tiene sentido pedir más resultados de los que existen.
    n_resultados = min(top_k, collection.count())

    resultado = collection.query(
        query_embeddings=[embedding],
        n_results=n_resultados,
        include=["documents", "metadatas", "distances"],
    )

    documents = resultado.get("documents", [[]])[0]
    metadatas = resultado.get("metadatas", [[]])[0]
    distances = resultado.get("distances", [[]])[0]

    recuperados: list[dict[str, Any]] = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        metadata = metadata or {}

        recuperados.append(
            {
                "text": document,
                "source": metadata.get("source", "desconocido"),
                "metadata": metadata,
                "distance": float(distance),
            }
        )

    return recuperados