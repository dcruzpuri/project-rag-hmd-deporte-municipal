"""
src/tsd/tag.py
Etiquetado semántico offline con LLM local (Ollama).
Se busca implementar una taxonomía cerrada (doc_category + tags + relevancia 0-1)
que sea utilizable para clasificación y recuperación de información, buscando
maximizar la precisión en la categorización y la fiabilidad de la información
recuperada respecto del contexto planteado.
"""
import json
import requests
from langchain_core.documents import Document
from config import OLLAMA_BASE_URL, GEN_PROVIDER, GEN_MODEL

PROMPT = """Clasifica este documento sobre deporte municipal de Madrid.
Devuelve SOLO JSON válido:
{"categoria": "tarifas|normativa|reservas|abonos|instalaciones|agenda",
 "tags": ["máx. 6 de: abono, piscina, reserva, tarifa, horario,
          empadronado, descuento, competición, instalación, precio"],
 "relevancia": 0.0-1.0}
Criterio de relevancia: utilidad para responder a preguntas como
'¿Cuánto cuesta el abono de piscina?', '¿Puedo reservar siendo no empadronado?',
tarifas, horarios y normas de uso.
Texto:
{text}"""

def _parse_json(texto: str) -> dict:
    texto = texto.strip()
    for prefijo in ("```json", "```"):
        if texto.startswith(prefijo):
            texto = texto[len(prefijo):]
    return json.loads(texto.strip().removesuffix("```"))

def etiquetar(documentos: list[Document]) -> list[Document]:
    """Etiqueta una vez por fuente y propaga a todos sus documentos."""
    por_fuente: dict[str, Document] = {}
    for doc in documentos:
        por_fuente.setdefault(doc.metadata["source"], doc)

    session = requests.Session()
    for fuente, doc in por_fuente.items():
        r = session.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": GEN_MODEL, "stream": False,
                  "messages": [{"role": "user",
                                "content": PROMPT.replace("{text}", doc.page_content[:6000])}]},
            timeout=120,
        )
        r.raise_for_status()
        d = _parse_json(r.json()["message"]["content"])
        meta = {
            "doc_category": d["categoria"],
            "tags": ";".join(d.get("tags", [])),   # str: Chroma no acepta listas [4]
            "relevancia_llm": float(d.get("relevancia", 0.0)),
        }
        for doc in documentos:                     # propagar a todas las páginas/filas
            if doc.metadata["source"] == fuente:
                doc.metadata.update(meta)
    return documentos