"""
src/embed.py (variante Ollama)
Convierte texto en vectores usando Ollama local.
"""


import requests
from config import EMBED_MODEL, EMBED_BATCH_SIZE, OLLAMA_BASE_URL
import json
from pathlib import Path

def _obtener_client():
    """No hay 'client' persistente con Ollama: usamos requests directo."""
    return requests.Session()


def embeddear(
    textos: list[str],
    model: str | None = None,
) -> list[list[float]]:
    """
    Convierte una lista de textos en vectores de 768 dims.

    Usa el endpoint batch /api/embed de Ollama (acepta varios inputs a la vez).
    """
    session = _obtener_client()
    model = model or EMBED_MODEL
    url = f"{OLLAMA_BASE_URL}/api/embed"

    vectores: list[list[float]] = []
    for inicio in range(0, len(textos), EMBED_BATCH_SIZE):
        batch = textos[inicio:inicio + EMBED_BATCH_SIZE]
        resp = session.post(
            url,
            json={"model": model, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        vectores.extend(data["embeddings"])

    return vectores


def embeddear_consulta(pregunta: str, model: str | None = None) -> list[float]:
    """
    Embed de una sola consulta (fase online / retrieval).
    MISMO modelo que se usó al indexar [10].
    """
    session = _obtener_client()
    url = f"{OLLAMA_BASE_URL}/api/embed"

    resp = session.post(
        url,
        json={"model": model or EMBED_MODEL, "input": [pregunta]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def exportar_json(
    chunks: list,
    ruta: str = "output/embeddings.json",
) -> None:
    """Igual que antes: persistir para inspección / debug."""
    
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)