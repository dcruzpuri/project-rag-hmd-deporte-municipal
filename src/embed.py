"""
src/embed.py
Convierte texto en vectores. Soporta 3 proveedores:
  - ollama       (offline)  -> POST /api/embed
  - huggingface  (online)   -> sentence-transformers
  - google       (online)   -> REST batchEmbedContents
Condición RAG: el índice y la consulta deben usar el MISMO modelo [5].
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

import requests

from config import (
    EMBED_PROVIDER, EMBED_MODEL, EMBED_BATCH_SIZE, OLLAMA_BASE_URL,
    HF_EMBED_MODEL, HF_DEVICE, HF_TOKEN,
    GOOGLE_API_KEY, GOOGLE_EMBED_MODEL,
)

# Caché: SentenceTransformer se carga una sola vez por modelo (evita `global`)
_HF_CACHE: dict[str, SentenceTransformer] = {}


def _embed_ollama(textos: list[str], model: str | None = None) -> list[list[float]]:
    """Ollama local: endpoint batch /api/embed (acepta varios inputs a la vez)."""
    session = requests.Session()
    url = f"{OLLAMA_BASE_URL}/api/embed"
    vectores: list[list[float]] = []
    for inicio in range(0, len(textos), EMBED_BATCH_SIZE):
        batch = textos[inicio:inicio + EMBED_BATCH_SIZE]
        resp = session.post(url, json={"model": model or EMBED_MODEL, "input": batch}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "embeddings" not in data:
            raise RuntimeError(
                "Ollama no devolvió 'embeddings': el endpoint batch /api/embed "
                "requiere Ollama >= 0.9. Actualizad Ollama o usad "
                "EMBED_PROVIDER=huggingface."
            )
        vectores.extend(data["embeddings"])
    return vectores


def _embed_huggingface(textos: list[str], model: str | None = None) -> list[list[float]]:
    """HuggingFace vía sentence-transformers (modelo descargado del HF Hub)."""
    nombre = model or HF_EMBED_MODEL
    if nombre not in _HF_CACHE:  # solo si el proveedor es huggingface
        kwargs: dict = {"device": HF_DEVICE}
        if HF_TOKEN:  # solo para modelos gated
            kwargs["token"] = HF_TOKEN
        _HF_CACHE[nombre] = SentenceTransformer(nombre, **kwargs)
    # normalize_embeddings=True: vectores unitarios, coherente con la métrica de coseno de Chroma
    vecs = _HF_CACHE[nombre].encode(textos, batch_size=EMBED_BATCH_SIZE,
                                    normalize_embeddings=True)
    return vecs.tolist()


def _embed_google(textos: list[str], model: str | None = None) -> list[list[float]]:
    """Google Gemini vía REST (requiere GOOGLE_API_KEY)."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY no está definida en .env")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model or GOOGLE_EMBED_MODEL}:batchEmbedContents")
    headers = {"x-goog-api-key": GOOGLE_API_KEY}
    vectores: list[list[float]] = []
    for inicio in range(0, len(textos), EMBED_BATCH_SIZE):
        batch = textos[inicio:inicio + EMBED_BATCH_SIZE]
        body = {"contents": [{"parts": [{"text": t}]} for t in batch]}
        resp = requests.post(url, headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        # La API devuelve "values" (o "value" en algunas versiones) por embedding
        for r in resp.json()["embeddings"]:
            emb = r.get("values") or r.get("value")
            if emb is None:
                raise RuntimeError(
                    "Google API no devolvió 'values' ni 'value' por embedding: "
                    f"respuesta parcial: {r!r}"
                )
            vectores.extend(emb)
    return vectores


# Dispatch por proveedor: añadir un proveedor nuevo = añadir una función y una línea
_PROVIDERS = {"ollama": _embed_ollama, "huggingface": _embed_huggingface, "google": _embed_google}

def _normalizar(vectores: list[list[float]]) -> list[list[float]]:
    E = np.asarray(vectores, dtype=np.float32)
    normas = np.linalg.norm(E, axis=1, keepdims=True)
    normas[normas == 0] = 1.0  # evitar división por cero
    return (E / normas).tolist()

def embeddear(textos: list[str], model: str | None = None) -> list[list[float]]:
    """Convierte una lista de textos en vectores usando el proveedor de EMBED_PROVIDER."""
    fn = _PROVIDERS.get(EMBED_PROVIDER)
    if fn is None:
        raise ValueError(f"EMBED_PROVIDER no soportado: {EMBED_PROVIDER!r}")
    return fn(textos, model)


def embeddear_consulta(pregunta: str, model: str | None = None) -> list[float]:
    """Embed de una sola consulta (fase online / retrieval). MISMO modelo que al indexar [5]."""
    return embeddear([pregunta], model)[0]


def exportar_json(chunks: list, embeddings: list[list[float]] | None = None,
                  ruta: str = "output/embeddings.json") -> None:
    """Persiste text+metadata+embedding para inspección/debug."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    registros = []
    for i, chunk in enumerate(chunks):
        reg = {"text": chunk.page_content, "metadata": chunk.metadata}
        if embeddings is not None and i < len(embeddings):
            reg["embedding"] = embeddings[i]
        registros.append(reg)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2, default=str)