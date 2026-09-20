"""
src/prompts.py

Construcción del prompt para el LLM en el pipeline RAG.

Responsabilidad única:
    contexto + pregunta -> prompt con bloques delimitados

Este módulo NO llama al LLM (eso es generate.py) ni recupera chunks
(eso es retrieve.py).
"""

from __future__ import annotations

# Fallback si config.py todavía no expone estas variables
try:
    from config import ABSTENTION_MESSAGE
except ImportError:
    ABSTENTION_MESSAGE = (
        "No dispongo de esa información en los documentos proporcionados."
    )


INSTRUCCIONES_SISTEMA = f"""Eres un asistente experto en deporte municipal del Ayuntamiento de Madrid.
Tu tarea es responder preguntas del usuario utilizando ÚNICAMENTE la información
contenida en el CONTEXTO proporcionado.

Reglas obligatorias:
1. Responde exclusivamente con información del CONTEXTO.
2. NO uses conocimiento externo ni inventes datos (fechas, precios, horarios, direcciones…).
3. Si el CONTEXTO no contiene información suficiente para responder, responde
   literalmente: "{ABSTENTION_MESSAGE}"
4. Cuando cites un hecho, menciona la fuente si aparece en el contexto.
5. Sé conciso y directo. No repitas la pregunta.
6. Responde en español."""


def formatear_contexto(chunks: list[dict]) -> str:
    """
    Convierte la lista de chunks recuperados en un bloque de contexto legible.

    Args:
        chunks: lista de dicts con 'text' y 'source'.

    Returns:
        Texto con cada fragmento delimitado, numerado y con su fuente.
    """
    if not chunks:
        return "(sin fragmentos recuperados)"

    fragmentos: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        fuente = chunk.get("source", "desconocido")
        texto = (chunk.get("text") or "").strip()
        fragmentos.append(
            f"--- Fragmento {i} (fuente: {fuente}) ---\n{texto}"
        )
    return "\n\n".join(fragmentos)


def construir_prompt(contexto: str, pregunta: str) -> str:
    """
    Ensambla el prompt final con bloques claros: instrucciones + contexto + pregunta.
    """
    pregunta_limpia = (pregunta or "").strip()
    return (
        f"{INSTRUCCIONES_SISTEMA}\n\n"
        f"=== CONTEXTO ===\n"
        f"{contexto.strip()}\n\n"
        f"=== PREGUNTA ===\n"
        f"{pregunta_limpia}\n\n"
        f"=== RESPUESTA ==="
    )


def construir_prompt_desde_chunks(chunks: list[dict], pregunta: str) -> str:
    """
    Atajo: formatea los chunks y construye el prompt en una sola llamada.
    Útil para logic.py y para tests.
    """
    return construir_prompt(formatear_contexto(chunks), pregunta)