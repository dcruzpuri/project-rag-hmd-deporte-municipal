"""project_break_rag: paquete principal para el pipeline RAG

Este paquete agrupa los bloques de construcción de un sistema de Generación con Recuperación Aumentada:
carga de documentos, segmentación, embeddings, almacenamiento vectorial, recuperación y generación.
Los submódulos se importan de manera ligera y no extraen dependencias pesadas hasta que sean necesarias. 
Esto permite importar el paquete principal sin cargar clientes de LLM o bases de datos vectoriales, 
y solo extraer esas dependencias cuando se accede a un submódulo específico.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Mapa de nombres públicos a los submódulos que los implementan.
_SUBMODULOS_LIGEROS: dict[str, str] = {
    "load": "load",
    "clean": "clean",
    "chunk": "chunk",
    "embed": "embed",
    "index": "index",
    "pipeline": "pipeline",
}

__all__ = [
    "__version__",
    "load",
    "clean",
    "chunk",
    "embed",
    "index",
    "pipeline",
]

if TYPE_CHECKING:
    from . import chunk, clean, embed, index, load, pipeline

def __getattr__(name: str):
    """Importa de manera ligera los submódulos cuando son accedidos por primera vez.

    Esto mantiene ``import project_break_rag.src``con un coste bajo en cuanto a proceso 
    y evitando importar dependencias pesadas (p.e. almacenes vectoriales o clientes de LLM) hasta que un
    componente específico sea solicitado.
    """
    if name in _SUBMODULOS_LIGEROS:
        module = importlib.import_module(
            f".{_SUBMODULOS_LIGEROS[name]}",
            __name__,
        )
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expone los módulos ligeros en ``dir()``"""
    return sorted(list(globals()) + __all__)
