"""
src/logic.py

Orquestador del pipeline RAG online.

Contrato público (NO cambiar sin avisar a David/Streamlit):

    responder(pregunta, top_k=None) -> dict
        {
            "respuesta": str,
            "chunks": list[dict],
            "fuentes": list[str],
            "contexto": str,
            "metrics": {...},
            "abstained": bool,
            "error": str | None,
        }

    rag_ask(consulta) -> str
        Envoltorio para el módulo de Agentes (devuelve solo la respuesta).

Pipeline:
    pregunta -> validación -> retrieval -> guardrail de evidencia
             -> prompt -> generación -> dict de resultado
"""

from __future__ import annotations

from typing import Any

from config import (
    ABSTENTION_MESSAGE,
    GEN_MODEL,
    GEN_PROVIDER,
    MAX_CHUNKS,
    TOP_K,
)
from src.generate import generar
from src.logging_utils import log_error, log_info, log_consulta, medir_tiempo
from src.prompts import construir_prompt_desde_chunks
from src.retrieve import recuperar


# Umbral de distancia coseno por encima del cual consideramos "sin evidencia".
# Basado en las pruebas de retrieval: in-corpus 0.43–0.55, fuera de corpus >0.73.
# 0.65 deja margen para preguntas ambiguas sin forzar abstención en casos límite.
UMBRAL_ABSTENCION: float = 0.65


def _extraer_fuentes(chunks: list[dict]) -> list[str]:
    """Devuelve la lista única de fuentes preservando el orden de aparición."""
    vistas: list[str] = []
    for c in chunks:
        s = c.get("source")
        if s and s not in vistas:
            vistas.append(s)
    return vistas


def _validar_pregunta(pregunta: str) -> str | None:
    """Devuelve un mensaje de error si la pregunta no es válida, o None si lo es."""
    if not isinstance(pregunta, str) or not pregunta.strip():
        return "La pregunta no puede estar vacía."
    if len(pregunta) > 2000:
        return "La pregunta es demasiado larga (máx. 2000 caracteres)."
    return None


def _resultado_error(mensaje: str) -> dict[str, Any]:
    """Respuesta uniforme cuando algo falla antes de llamar al LLM."""
    return {
        "respuesta": "",
        "chunks": [],
        "fuentes": [],
        "contexto": "",
        "metrics": {},
        "abstained": False,
        "error": mensaje,
    }


def _hay_evidencia(chunks: list[dict], umbral: float = UMBRAL_ABSTENCION) -> bool:
    """
    True si al menos un chunk está por debajo del umbral de distancia.
    Si no hay chunks, no hay evidencia.
    """
    if not chunks:
        return False
    mejor_dist = min(c.get("distance", 1.0) for c in chunks)
    return mejor_dist <= umbral


def responder(pregunta: str, top_k: int | None = None) -> dict[str, Any]:
    """
    Pipeline RAG completo: pregunta -> retrieval -> prompt -> LLM -> respuesta.

    Args:
        pregunta: consulta del usuario.
        top_k: número de chunks a recuperar (por defecto TOP_K de config).

    Returns:
        dict con respuesta, chunks, fuentes, contexto, metrics, abstained y error.
    """
    # 1) Validación pre-LLM
    error = _validar_pregunta(pregunta)
    if error:
        log_error(error)
        return _resultado_error(error)

    k = top_k if top_k is not None else TOP_K
    if k <= 0:
        return _resultado_error("top_k debe ser mayor que 0.")

    pregunta = pregunta.strip()
    metricas: dict[str, float] = {}

    # 2) Retrieval
    try:
        with medir_tiempo("retrieval", metricas):
            chunks = recuperar(pregunta, top_k=k)
    except Exception as e:
        log_error(f"Fallo en retrieval: {e}")
        return _resultado_error(f"Error recuperando contexto: {e}")

    # 3) Guardrail de evidencia: sin chunks o distancias altas -> abstención sin LLM
    if not _hay_evidencia(chunks):
        log_info(
            f"Sin evidencia suficiente para: {pregunta!r} "
            f"(top-k={k}, chunks={len(chunks)})"
        )
        return {
            "respuesta": ABSTENTION_MESSAGE,
            "chunks": chunks,
            "fuentes": _extraer_fuentes(chunks),
            "contexto": "",
            "metrics": {**metricas, "top_k": k, "n_chunks": 0},
            "abstained": True,
            "error": None,
        }

    # 4) Limitar chunks que entran al prompt (MAX_CHUNKS)
    chunks_prompt = chunks[:MAX_CHUNKS]
    fuentes = _extraer_fuentes(chunks_prompt)

    # 5) Prompt con grounding
    prompt = construir_prompt_desde_chunks(chunks_prompt, pregunta)

    # 6) Generación
    try:
        with medir_tiempo("generation", metricas):
            respuesta = generar(prompt)
    except Exception as e:
        log_error(f"Fallo en generación: {e}")
        return _resultado_error(f"Error generando respuesta: {e}")

    # 7) Detección de abstención del LLM: si el modelo devuelve el mensaje literal,
    #    lo marcamos como abstención (aunque haya pasado el guardrail de distancia).
    abstained = ABSTENTION_MESSAGE.lower() in respuesta.lower()

    # 8) Logging estructurado
    log_consulta(
        pregunta=pregunta,
        top_k=k,
        n_chunks=len(chunks_prompt),
        modelo=f"{GEN_PROVIDER}:{GEN_MODEL}",
        metricas=metricas,
        abstained=abstained,
    )

    return {
        "respuesta": respuesta,
        "chunks": chunks_prompt,
        "fuentes": fuentes,
        "contexto": "\n\n".join(c.get("text", "") for c in chunks_prompt),
        "metrics": {
            **metricas,
            "top_k": k,
            "n_chunks": len(chunks_prompt),
            "model": GEN_MODEL,
        },
        "abstained": abstained,
        "error": None,
    }


def rag_ask(consulta: str) -> str:
    """
    Envoltorio para el módulo de Agentes: devuelve solo el texto de la respuesta.

    Si hubo error, devuelve el mensaje de abstención para no romper la cadena.
    """
    resultado = responder(consulta)
    if resultado.get("error"):
        return ABSTENTION_MESSAGE
    return resultado.get("respuesta", "")