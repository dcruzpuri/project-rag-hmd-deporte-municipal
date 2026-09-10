## Integración en pipeline.py 

```python 
from .tsd/tag     import etiquetar
from .tsd/scoring import puntuar
from .tsd/dedup   import deduplicar

# 2. CLEAN
documentos = limpiar(documentos)

# 2.5 TAG (por fuente, antes de trocear)
documentos = etiquetar(documentos)

# 3. CHUNK  → hereda doc_category, tags, relevancia_llm [1]
chunks = trocear(documentos)

# 4. EMBED
embeddings = embeddear([c.page_content for c in chunks])

# 4.5 SCORING + DEDUP
chunks = puntuar(chunks, embeddings)
chunks, embeddings = deduplicar(chunks, embeddings)

# 5. INDEX
```

`semantic_score` es `float` y `tags` es `str`, así que pasan el saneador de index.py sin problemas.

## Retrieval con filtrado por doc_category

```python 
from .embed import embeddear_consulta
from .index import obtener_cliente_chroma
from config import COLLECTION_NAME

INTENCIONES = {"abono": "abonos", "reserv": "reservas",
               "cuesta": "tarifas", "precio": "tarifas", "tarifa": "tarifas",
               "horario": "agenda", "instalaciones": "piscina"}

def clasificar_intencion(pregunta: str) -> str | None:
    p = pregunta.lower()
    for clave, categoria in INTENCIONES.items():
        if clave in p:
            return categoria
    return None

def buscar(pregunta: str, categoria: str | None = None, top_k: int = 5):
    vec = embeddear_consulta(pregunta)  # mismo modelo que al indexar 
    res = obtener_cliente_chroma().get_collection(COLLECTION_NAME).query(
        query_embeddings=[vec],
        where={"doc_category": categoria} if categoria else None,
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    # rerank por semantic_score antes de pasar al LLM generador
    docs = sorted(zip(res["documents"], res["metadatas"]),
                  key=lambda x: x[1].get("semantic_score", 0.0), reverse=True)
    return docs[:top_k]
``` 

### Notas de ajuste

Umbral 0.93 es el punto de partida para all-minilm-l6-v2 `config.py`
Subir a 0.95 si se dedup demasiado, bajar a 0.90 si queda redundancia (p. ej. PreciosPublicos2026.pdf vs Tarifas_deportivas.pdf).
top_k: con el filtro por categoría y el rerank por semantic_score, se podría subir el TOP_K de 3 a 5 sin perder precisión.
Verificación: el pipeline ya devuelve métricas `pipeline.py`
Añadir num_chunks_post_dedup al dict para ver cuántos chunks descarta la dedup en cada ejecución.
Si el LLM devuelve JSON malformado, _parse_json fallará: 0con **llama3.1** (`config.py`) suele ir bien, 
pero si hay fallos, añadir un try/except que asigne doc_category = "instalaciones" por defecto.
Quiero probar **llama4:scout** de Ollama.