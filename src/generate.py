"""
src/generate.py

Capa de generación: envía un prompt al LLM y devuelve la respuesta en texto.

Responsabilidad única:
    prompt -> LLM -> respuesta (str)

Este módulo NO recupera chunks (retrieve.py) ni construye el prompt
(prompts.py) ni orquesta el pipeline (logic.py).
"""

from __future__ import annotations

from google import genai
from google.genai import types

from config import (
    GEN_PROVIDER,
    GEN_MODEL,
    GOOGLE_API_KEY,
)

# Temperatura: si config.py no la define todavía, usamos 0.2 (apegado al contexto)
try:
    from config import GEN_TEMPERATURE
except ImportError:
    GEN_TEMPERATURE = 0.2


def _generar_google(prompt: str) -> str:
    """Generación con Gemini vía SDK google-genai."""
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY no está definida en .env. "
            "Configúrala antes de generar respuestas."
        )
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=GEN_TEMPERATURE),
    )
    texto = (response.text or "").strip()
    if not texto:
        raise RuntimeError("Gemini devolvió una respuesta vacía.")
    return texto


# Dispatch por proveedor: añadir uno nuevo = una función + una línea aquí
_PROVEEDORES = {
    "google": _generar_google,
}


def generar(prompt: str, provider: str | None = None) -> str:
    """
    Envía un prompt al LLM y devuelve la respuesta como texto plano.

    Args:
        prompt: prompt completo (instrucciones + contexto + pregunta).
        provider: override del proveedor. Por defecto, GEN_PROVIDER de config.

    Returns:
        Respuesta del LLM como string (nunca vacía; si lo fuera, lanza error).

    Raises:
        ValueError: si el proveedor no está soportado o el prompt está vacío.
        RuntimeError: si la API falla o devuelve respuesta vacía.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("El prompt no puede estar vacío.")

    proveedor = provider or GEN_PROVIDER
    fn = _PROVEEDORES.get(proveedor)
    if fn is None:
        raise ValueError(
            f"GEN_PROVIDER no soportado: {proveedor!r}. "
            f"Soportados: {list(_PROVEEDORES)}"
        )
    return fn(prompt.strip())