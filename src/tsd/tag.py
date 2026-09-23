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
from typing import Any

import requests
from langchain_core.documents import Document

from transformers import pipeline

from config import (
    OLLAMA_BASE_URL,
    TAG_PROVIDER, TAG_MODEL, HF_TAG_MODEL, GOOGLE_TAG_MODEL,
    HF_DEVICE, HF_TOKEN, GOOGLE_API_KEY, TAG_MAX_TOKENS,
)

PROMPT = """Clasifica este documento sobre deporte municipal de Madrid.
Devuelve SOLO JSON válido:
{"categoria": "tarifas|normativa|reservas|abonos|instalaciones|agenda",
"tags": ["máx. 8 de: abono, piscina, reserva, tarifa, horario,
empadronado, descuento, competición, instalación, precio, cancelación,
devolución, accesibilidad, aire_libre, inscripción, temporada"],
"relevancia": 0.0-1.0}
Categorías (elige la intención DOMINANTE del documento):
- tarifas: precios y coste de uso (abonos, entradas, bonificaciones).
- reservas: intención de reservar, cancelar o devolver: quién puede
  reservar, hasta cuándo se puede cancelar, cuánto cuesta cancelarlo y
  devoluciones. Un texto que REGULA la reserva/cancelación (aunque su tono
  sea normativo) va a reservas, no a normativa.
- normativa: reglas de uso generales (acceso, quejas, supervisión),
  cuando NO giran en torno a reservar/cancelar/devolver.
- abonos: qué es un abono y cómo se adquiere/renewa; instalaciones: catálogos
  de instalaciones; agenda: eventos y temporadas.
Ejemplos de categoría: '¿Puedo reservar siendo no empadronado?', '¿Qué pasa
si cancelo con coste?', '¿Hasta qué hora se puede cancelar sin coste?' -> reservas.
Criterio de relevancia: utilidad para responder a preguntas como
'¿Cuánto cuesta el abono de piscina?', '¿Puedes reservar siendo no empadronado?',
tarifas, horarios y normas de uso.
Texto:
{text}"""

# Taxonomía cerrada: el LLM debe elegir de estas listas; el parse filtra a
# estos valores (la metadata del índice es filtrable en la fase online).
CATEGORIAS_VALIDAS = frozenset({
    "tarifas", "normativa", "reservas", "abonos", "instalaciones", "agenda",
})
TAGS_VALIDAS = frozenset({
    "abono", "piscina", "reserva", "tarifa", "horario",
    "empadronado", "descuento", "competición", "instalación", "precio",
    "cancelación", "devolución", "accesibilidad", "aire_libre",
    "inscripción", "temporada",
})

# Modelo de tagging según proveedor (se resuelve una sola vez, al importar)
_MODELOS_TAG = {
    "ollama": TAG_MODEL,
    "huggingface": HF_TAG_MODEL,
    "google": GOOGLE_TAG_MODEL,
}

_HF_GEN = None  # caché: el pipeline de HF es caro de cargar, se reutiliza


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
    global _HF_GEN
    if _HF_GEN is None:  # carga perezosa + caché
        kwargs = {"device": HF_DEVICE}
        if HF_TOKEN:  # solo para modelos gated
            kwargs["token"] = HF_TOKEN
        _HF_GEN = pipeline(
            "text-generation",
            model=model,
            use_fast=True,
            trust_remote_code=False,
            model_kwargs={},
        )
    out = _HF_GEN(
        prompt, 
        do_sample=False, 
        return_full_text=False
    )
    return out[0]["generated_text"]


def _chat_google(prompt: str, model: str) -> str:
    """Google Gemini (online): REST generateContent (requiere GOOGLE_API_KEY)."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY no está definida en .env")
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": GOOGLE_API_KEY},
        json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
              "generationConfig": {
                  "temperature": 0.2,
                  "maxOutputTokens": TAG_MAX_TOKENS,
                  # JSON puro, sin bloque ```json ni prosa alrededor.
                  "responseMimeType": "application/json",
              }},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


# Dispatch por proveedor: añadir uno nuevo = añadir una función y una línea
_CHAT = {"ollama": _chat_ollama, "huggingface": _chat_huggingface, "google": _chat_google}


def _parse_json(texto: str) -> dict[str, Any]:
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
        prompt = PROMPT.replace("{text}", doc.page_content[:8000])  # DOCUMENTAR: el tag solo ve los primeros 6000 caracterers del documento fuente
        try:
            d = _parse_json(fn_chat(prompt, modelo))
            if not isinstance(d, dict):
                raise ValueError("el LLM devolvió un JSON que no es un objeto")
        except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError) as exc:
            # LLM con JSON malformado: categoría por defecto en vez de romper el pipeline
            print(f"[TAG]   {fuente}: JSON malformado ({type(exc).__name__}: {exc}) "
                  f"-> 'instalaciones' por defecto")
            d = {"categoria": "instalaciones", "tags": [], "relevancia": 0.5}
        # Taxonomía cerrada: se filtra a los valores conocidos (la metadata del
        # índice debe ser filtrable en la fase online; un tag libre la rompería).
        categoria = d.get("categoria", "instalaciones")
        if categoria not in CATEGORIAS_VALIDAS:
            print(
                f"[TAG]   {fuente}: categoría {categoria!r} "
                f"fuera de taxonomía -> 'instalaciones'"
            )
            categoria = "instalaciones"
        tags = [t for t in d.get("tags", []) if t in TAGS_VALIDAS]
        meta = {
            "doc_category": categoria,
            "tags": ";".join(tags),  # str: Chroma no acepta listas (lectura humana)
            "relevancia_llm": float(d.get("relevancia", 0.0)),
            # Booleanos por tag (sparse: solo se escribe el tag presente).
            # Chroma `where` soporta booleanos; la clave ausente no matchea,
            # así la fase online (--query) filtra p. ej. con
            # where={"tag_piscina": True} o where={"$and": [...]}.
            **{f"tag_{t}": True for t in tags},
        }
        for otro in documentos:  # propagar a todas las páginas/filas de la fuente
            if otro.metadata.get("source") == fuente:
                otro.metadata.update(meta)
        print(f"[TAG]   {fuente} -> {meta['doc_category']} (relevancia {meta['relevancia_llm']:.2f})")

    print(f"[TAG]   {len(por_fuente)} fuentes etiquetadas en {time.time() - t0:.1f}s "
          f"(proveedor: {TAG_PROVIDER}, modelo: {modelo})")
    return documentos

