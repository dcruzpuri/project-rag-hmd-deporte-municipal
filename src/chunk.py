"""
src/chunk.py
Divide documentos en fragmentos pequeños y homogéneos.
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
        documentos:   lista de Document (ya limpios).
        chunk_size:   override temporal (default: config.CHUNK_SIZE).
        chunk_overlap: override temporal (default: config.CHUNK_OVERLAP).
    """
    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP
    
    # Aceptar string plano (tests / eval) y envolver en Document o lista de Document (pipeline)
    if isinstance(documentos, str):
        documentos = [Document(page_content=documentos, metadata={"source": "inline"})]


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        # Orden de separadores: párrafo → frase → espacio → letra
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documentos:
        doc_chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(doc_chunks):
            # Preservar metadata original del documento padre
            chunk.metadata = {**doc.metadata, "chunk_index": i}
            chunks.append(chunk)
    return chunks