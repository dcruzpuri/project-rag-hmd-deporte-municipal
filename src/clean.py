"""
src/clean.py
Quita ruido antes de trocear para producir chunks más homogéneos.
"""

import re

from langchain_core.documents import Document


def _normalizar_page_content(texto: str) -> str:
    """
    Reglas de limpieza (aplica todas en orden):
      1. Saltos de línea dobles o más → uno solo.
      2. Espacios de más (tabs, 3+ espacios) → uno.
      3. Líneas vacías repetidas → eliminar.
      4. Espacios al inicio/final de cada línea.
      5. BOM / caracteres de control invisibles.
    """
    # 1
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    # 2
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    # 3
    texto = re.sub(r"(\n\s*){2,}", "\n\n", texto)
    # 4
    texto = "\n".join(linea.strip() for linea in texto.split("\n"))
    # 5
    texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texto)
    # Eliminar BOM
    texto = texto.lstrip("\ufeff")
    return texto.strip()


def limpiar(documentos: list[Document] | str) -> list[Document] | str:
    """
    Recibe documentos crudos o un string, devuelve el texto normalizado.
    No modifica metadata.
    """
    # Si viene un string plano, normaliza y devuelve string (según docstring y firma)
    if isinstance(documentos, str):
        return _normalizar_page_content(documentos)
    limpios: list[Document] = []
    for doc in documentos:
        limpios.append(Document(
            page_content=_normalizar_page_content(doc.page_content),
            metadata=dict(doc.metadata),  # copia shallow de metadatos
        ))
    return limpios