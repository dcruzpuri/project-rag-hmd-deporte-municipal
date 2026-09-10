"""
src/load.py
Abstrae el formato: PDF, TXT, MD, CSV -> lista de LangChain Documents.
"""

import csv
import os
from pathlib import Path
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def _load_pdf(path: str) -> list[Document]:
    loader = PyPDFLoader(path)
    return loader.load()


def _load_text(path: str) -> list[Document]:
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def _load_csv(path: str) -> list[Document]:
    """
    CSV de eventos: una fila = un documento.
    Convierte cada fila en un texto plano legible.
    """
    docs: list[Document] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            docs.append(Document(
                page_content=text,
                metadata={"source": os.path.basename(path), "row": i},
            ))
    return docs

# Registro extensión: - loader
Loader = Callable[[str], list[Document]]
_LOADERS: dict[str, Loader] = {
    ".pdf": _load_pdf,
    ".txt": _load_text,
    ".md":  _load_text,
    ".csv": _load_csv,
}


def cargar_archivos(rutas: list[str] | str) -> list[Document]:
    """
    Punto de entrada público.

    Params:
        rutas: una ruta o lista de rutas a archivos/carpetas.

    Returns:
        Lista de Document (page_content + metadata.source).
    """
    if isinstance(rutas, str):
        rutas = [rutas]

    archivos: list[str] = []
    for ruta in rutas:
        p = Path(ruta)
        if p.is_dir():
            archivos.extend(
                str(f) for f in sorted(p.rglob("*"))
                if f.is_file() and f.suffix.lower() in _LOADERS
            )
        elif p.is_file() and p.suffix.lower() in _LOADERS:
            archivos.append(str(p))

    documentos: list[Document] = []
    for archivo in archivos:
        ext = Path(archivo).suffix.lower()
        loader_fn = _LOADERS[ext]
        docs = loader_fn(archivo)
        # Asegurar metadata.source si el loader no la puso
        for d in docs:
            d.metadata.setdefault("source", os.path.basename(archivo))
        documentos.extend(docs)

    return documentos