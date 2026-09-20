"""
src/logging_utils.py

Utilidades de logging estructurado para el pipeline RAG online.

Responsabilidad:
    Centralizar el formato de logs y las métricas de tiempo.

Este módulo NO contiene lógica RAG. Solo formatea y emite mensajes.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any

_PREFIJO = "[RAG]"


def log(mensaje: str, nivel: str = "INFO") -> None:
    """Log simple con prefijo y timestamp."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {_PREFIJO} [{nivel}] {mensaje}")


def log_info(mensaje: str) -> None:
    log(mensaje, "INFO")


def log_warning(mensaje: str) -> None:
    log(mensaje, "WARN")


def log_error(mensaje: str) -> None:
    log(mensaje, "ERROR")


@contextmanager
def medir_tiempo(nombre: str, metricas: dict[str, float] | None = None):
    """
    Context manager para medir tiempos y registrarlos.

    Uso:
        metricas = {}
        with medir_tiempo("retrieval", metricas):
            chunks = recuperar(pregunta)
        # metricas["retrieval"] contiene los segundos
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if metricas is not None:
            metricas[nombre] = round(dt, 4)


def log_consulta(
    pregunta: str,
    top_k: int,
    n_chunks: int,
    modelo: str,
    metricas: dict[str, float],
    abstained: bool = False,
) -> None:
    """
    Log estructurado de una consulta RAG completa.

    Formato JSON con:
        - pregunta
        - top_k
        - n_chunks (chunks que entraron al prompt)
        - modelo (proveedor:modelo)
        - abstained (True si el sistema se abstuvo)
        - t_* (tiempos por fase: t_retrieval, t_generation, …)

    Útil para el informe y para la tabla de métricas de Streamlit.
    """
    payload: dict[str, Any] = {
        "pregunta": pregunta,
        "top_k": top_k,
        "n_chunks": n_chunks,
        "modelo": modelo,
        "abstained": abstained,
        **{f"t_{k}": v for k, v in metricas.items()},
    }
    log(json.dumps(payload, ensure_ascii=False))