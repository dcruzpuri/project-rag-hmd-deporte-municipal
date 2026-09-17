"""
src/load.py
Abstrae el formato: PDF, TXT, MD, CSV -> lista de LangChain Documents.
El CSV no se convierte aquí: ``csv_transform.py`` aplica el consejo del
advisor de ``csv_advisor.py`` (entidad, grupo o fila por documento).
"""

import hashlib
import os
from pathlib import Path
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from .csv_transform import transform_csv


_LOTES_HASH = 1 << 20  # 1 MiB por lectura en el hash


def _hash_file(ruta: str) -> str:
    """SHA-256 hex del contenido binario crudo del archivo (en streaming).

    Identifica unívocamente al archivo de origen: estable entre indexaciones,
    detecta contenido nuevo en una descarga repetida del mismo nombre. Se lee
    en lotes para no cargar en memoria los archivos grandes.
    """
    digest = hashlib.sha256()
    with open(ruta, "rb") as f:
        while True:
            bloque = f.read(_LOTES_HASH)
            if not bloque:
                break
            digest.update(bloque)
    return digest.hexdigest()


def _normalizar_source(documentos: list[Document], ruta: str) -> list[Document]:
    """source = os.path.basename(ruta) SIEMPRE (coherencia de metadata por clave
    de fuente: dedup, ids y matching de fuente_esperada en el eval), más
    file_hash = SHA-256 del archivo crudo (identifica unívocamente el origen,
    estable entre indexaciones).

    TextLoader/PyPDFLoader fijan la ruta completa (con data\\ en Windows);
    si el loader ya dejó un source limpio (p. ej. los CSV), no se toca.
    """
    file_hash = _hash_file(ruta)
    for d in documentos:
        d.metadata["source"] = os.path.basename(ruta)
        d.metadata["file_hash"] = file_hash
    return documentos


def _load_pdf(path: str) -> list[Document]:
    loader = PyPDFLoader(path)
    return _normalizar_source(loader.load(), path)


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
    return _normalizar_source(loader.load(), path)


def _load_csv(path: str) -> list[Document]:
    """
    CSV de eventos: csv_advisor.py decide el tratamiento según la estructura
    (entidad / grupo de hechos / texto plano) y csv_transform.py lo aplica.
    Aquí solo delegamos: load.py es un lector, no inventa transformaciones.
    """
    return _normalizar_source(transform_csv(path), path)

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
        Lista de Document (page_content + metadata.source + metadata.file_hash).
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
        # Asegurar metadata.source y metadata.file_hash si el loader no los puso
        file_hash = _hash_file(archivo)
        for d in docs:
            d.metadata.setdefault("source", os.path.basename(archivo))
            d.metadata.setdefault("file_hash", file_hash)
        documentos.extend(docs)

    return documentos