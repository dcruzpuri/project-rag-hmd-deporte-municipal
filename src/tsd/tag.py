"""
src/tsd/tag.py
Etiquetado semántico offline con LLM (interruptor TAG: Ollama / HuggingFace / Google).
Taxonomía cerrada (doc_category + tags + relevancia 0-1) utilizable para clasificación
y recuperación, buscando maximizar la precisión en la categorización y la fiabilidad
de la información recuperada.
Se etiqueta una vez por fuente y se propaga a todos sus documentos.
"""
import json
import time

import requests
from langchain_core.documents import Document

from transformers import pipeline

from config import (
    OLLAMA_BASE_URL,
    TAG_PROVIDER, TAG_MODEL, HF_TAG_MODEL, GOOGLE_TAG_MODEL,
    HF_DEVICE, HF_TOKEN, GOOGLE_API_KEY,
)

PROMPT = """Clasifica este documento sobre deporte municipal de Madrid.
Devuelve SOLO JSON válido:
{"categoria": "tarifas|normativa|reservas|abonos|instalaciones|agenda",
"tags": ["máx. 6 de: abono, piscina, reserva, tarifa, horario,
empadronado, descuento, competición, instalación, precio"],
"relevancia": 0.0-1.0}
Criterio de relevancia: utilidad para responder a preguntas como
'¿Cuánto cuesta el abono de piscina?', '¿Puedes reservar siendo no empadronado?',
tarifas, horarios y normas de uso.
Texto:
{text}"""

# Modelo de tagging según proveedor (se resuelve una sola vez, al importar)
_MODELOS_TAG = {
    "ollama": TAG_MODEL,
    "huggingface": HF_TAG_MODEL,
    "google": GOOGLE_TAG_MODEL,
}

_hf_gen = None  # caché: el pipeline de HF es caro de cargar, se reutiliza


def _chat_ollama(prompt: str, model: str) -> str:
    """Ollama local (offline): POST /api/chat."""
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": model, "stream": False,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def _chat_huggingface(prompt: str, model: str) -> str:
    """HuggingFace (online): pipeline text-generation de transformers."""
    global _hf_gen
    if _hf_gen is None:  # carga perezosa + caché
        kwargs = {"device": HF_DEVICE}
        if HF_TOKEN:  # solo para modelos gated
            kwargs["token"] = HF_TOKEN
        _hf_gen = pipeline("text-generation", model=model, **kwargs)
    out = _hf_gen(prompt, do_sample=False, return_full_text=False)
    return out[0]["generated_text"]


def _chat_google(prompt: str, model: str) -> str:
    """Google Gemini (online): REST generateContent (requiere GOOGLE_API_KEY)."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY no está definida en .env")
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": GOOGLE_API_KEY},
        json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256}},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


# Dispatch por proveedor: añadir uno nuevo = añadir una función y una línea
_CHAT = {"ollama": _chat_ollama, "huggingface": _chat_huggingface, "google": _chat_google}


def _parse_json(texto: str) -> dict:
    """Extrae el JSON de la respuesta del LLM (tolera bloques ```json y prosa)."""
    inicio, fin = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fin <= inicio:
        raise json.JSONDecodeError("sin JSON en la respuesta", texto, 0)
    return json.loads(texto[inicio:fin + 1])


def etiquetar(documentos: list[Document]) -> list[Document]:
    """Etiqueta una vez por fuente y propaga a todos sus documentos."""
    por_fuente: dict[str, Document] = {}
    for i, doc in enumerate(documentos):
        fuente = doc.metadata.get("source", f"doc_{i}")
        por_fuente.setdefault(fuente, doc)

    fn_chat = _CHAT.get(TAG_PROVIDER)
    if fn_chat is None:
        raise ValueError(f"TAG_PROVIDER no soportado: {TAG_PROVIDER!r}")
    modelo = _MODELOS_TAG[TAG_PROVIDER]

    t0 = time.time()
    for fuente, doc in por_fuente.items():
        prompt = PROMPT.replace("{text}", doc.page_content[:6000])  # DOCUMENTAR: el tag solo ve los primeros 6000 caracterers del documento fuente
        try:
            d = _parse_json(fn_chat(prompt, modelo))
            if not isinstance(d, dict):
                raise ValueError("el LLM devolvió un JSON que no es un objeto")
        except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError):
            # LLM con JSON malformado: categoría por defecto en vez de romper el pipeline
            print(f"[TAG]   {fuente}: JSON malformado -> 'instalaciones' por defecto")
            d = {"categoria": "instalaciones", "tags": [], "relevancia": 0.5}
        meta = {
            "doc_category": d.get("categoria", "instalaciones"),
            "tags": ";".join(d.get("tags", [])),  # str: Chroma no acepta listas
            "relevancia_llm": float(d.get("relevancia", 0.0)),
        }
        for otro in documentos:  # propagar a todas las páginas/filas de la fuente
            if otro.metadata.get("source") == fuente:
                otro.metadata.update(meta)
        print(f"[TAG]   {fuente} -> {meta['doc_category']} (relevancia {meta['relevancia_llm']:.2f})")

    print(f"[TAG]   {len(por_fuente)} fuentes etiquetadas en {time.time() - t0:.1f}s "
          f"(proveedor: {TAG_PROVIDER}, modelo: {modelo})")
    return documentos

