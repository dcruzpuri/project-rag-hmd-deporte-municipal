"""
src/chunk.py
Divide documentos en fragmentos pequeños y homogéneos.

Estrategia: "Level 2 – Recursive Character" del tutorial "Five levels of
text splitting": se corta primero en límites semánticos (párrafo → frase)
y solo después en espacios/caracteres. Es el punto de partida recomendado
("swiss army knife"); se valida con eval de retrieval
(scripts/eval_coherencia_chunks.py).
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def trocear(
    documentos: list[Document] | str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Divide documentos largos en chunks usando RecursiveCharacterTextSplitter.
    Enriquece cada chunk con chunk_index y chunk_size en metadata.

    Params:
        documentos:    lista de Document (ya limpios) o string plano (tests/eval).
        chunk_size:    override temporal (default: config.CHUNK_SIZE).
        chunk_overlap: override temporal (default: config.CHUNK_OVERLAP).
    """
    # `is not None`: 0 es un valor válido para overlap y no es falsy
    size = chunk_size if chunk_size is not None else CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP

    # Aceptar string plano (tests / eval) y envolver en Document
    if isinstance(documentos, str):
        documentos = [Document(page_content=documentos, metadata={"source": "inline"})]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        # Orden de separadores: párrafo → frase → espacio → letra [3]
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []

    def _sin_trocear(doc: Document) -> None:
        # Bypass por `chunking_hint` del advisor (csv_advisor): un documento ya
        # atómico (entidad) o corto (grupo light) no pasa por el splitter.
        doc.metadata = {**doc.metadata, "chunk_index": 0, "chunk_size": size}
        chunks.append(doc)

    for doc in documentos:
        hint = doc.metadata.get("chunking_hint", "normal_chunk")
        if hint == "no_chunk":
            _sin_trocear(doc)
        elif hint == "light_chunk" and len(doc.page_content) <= size:
            _sin_trocear(doc)
        else:
            doc_chunks = splitter.split_documents([doc])
            for i, chunk in enumerate(doc_chunks):
                # Puente TAG→CHUNK: hereda la metadata del documento padre
                # (doc_category, tags, relevancia_llm) + chunk_index secuencial
                # por documento [17]. chunking_hint del advisor también viaja.
                chunk.metadata = {**doc.metadata, "chunk_index": i, "chunk_size": size}
                chunks.append(chunk)
    return chunks