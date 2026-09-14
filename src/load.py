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


# En orden de preferencia. cp1252 mapea mejor que latin-1 los guiones y
# comillas tipicas de Windows (0x95, 0x96, 0x92...), pero algunos bytes
# (0x81, 0x8D, 0x8F, 0x90, 0x9D) no existen en cp1252: ahí entra latin-1.
_CODIFICACIONES: tuple[str, ...] = ("utf-8-sig", "cp1252", "latin-1")


def _detectar_encoding(path: str) -> str:
    """
    Detecta la encoding de un archivo probando en orden:
    utf-8-sig (maneja BOM), cp1252 (Windows), latin-1 (nunca falla).
    """
    raw = Path(path).read_bytes()
    for enc in _CODIFICACIONES:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    # Unreachable: latin-1 decodifica cualquier byte.
    return "latin-1"


def _load_text(path: str) -> list[Document]:
    loader = TextLoader(path, encoding=_detectar_encoding(path))
    return loader.load()


def _load_csv(path: str) -> list[Document]:
    """
    CSV de eventos: una fila = un documento.
    Convierte cada fila en un texto plano legible.
    """
    docs: list[Document] = []
    with open(path, newline="", encoding=_detectar_encoding(path)) as f:
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
            todos = sorted(p.rglob("*"))
            archivos.extend(str(f) for f in todos
                    if f.is_file() and f.suffix.lower() in _LOADERS)
            ignorados = [f.name for f in todos
                 if f.is_file() and f.suffix.lower() not in _LOADERS]
            if ignorados:
                print(f"[LOAD] ignorados (extensión no soportada): {ignorados}")
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