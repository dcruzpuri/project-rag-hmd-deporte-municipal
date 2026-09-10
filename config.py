"""
project_break_rag/config.py
Parámetros de todo el pipeline para ajustar aquí y no tocar módulos.
"""
from pathlib import Path

#  Rutas 
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
OUTPUT_DIR: Path = BASE_DIR / "output"
LOG_DIR: Path = BASE_DIR / "logs"

#  Chunking 
CHUNK_SIZE: int =  1000       # caracteres por chunk (700-1000 equilibrio)
CHUNK_OVERLAP: int = 100     # solapamiento (10-20% del chunking)


#  Scoring, deduplicación y etiquetado
TAG_SCORING_DEDUP: bool = True  # True: etiquetar → puntuar → deduplicar; False: solo etiquetar

#  Embeddings 
EMBED_PROVIDER: str = "ollama"                 # "ollama" o "google
EMBED_MODEL: str = "locusai/all-minilm-l6-v2"  # "locusai/all-minilm-l6-v2" para Ollama / "gemini-embedding-2" para Google Gemini
EMBED_DIM: int = 384                           # EMBED_DIM: int = 384 para "locusai/all-minilm-l6-v2" para Ollama / 3072 si usas Google Gemini
EMBED_BATCH_SIZE: int = 30                     # Ollama en local va bien con lotes menores
EXPORT_EMBEDDINGS: bool = True                 # True: exporta output/embeddings.json para inspección

#  Generación (LLM) 
GEN_PROVIDER: str = "ollama"                   # "ollama" / "google" / "huggingface" 
GEN_MODEL: str = "llama4:scout"                # llama3.1 con Ollama local / gemini-2.0-flash con Google Gemini
"""
    De la misma manera que los modelos de generación, para las operaciones de TAGGING y SCORING 
    está permitido utilizar otro modelo, incluso otro proveedor, que pudiera ser más adecuado en coste, evaluación,
    incluso pudiendo también hacerlo mediante modelo local (ollama) 
"""
TAG_PROVIDER: str = GEN_PROVIDER               # "ollama" / "google" / "huggingface" - aunque puede asumirse el mismo que GEN_PROVIDER
TAG_MODEL: str = GEN_PROVIDER                  # llama3.1 (ollama) / gemini-2.0-flash (google) / 
LLM_TEMPERATURE: float = 0.5
LLM_TIMEOUT: int = 60

#  Ollama
OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"  # servidor local ollama

#  ChromaDB 
CHROMA_DIR: str = str(OUTPUT_DIR / "chroma_db")      # carpeta persistente
COLLECTION_NAME: str = "deporte_municipal"         # una única colección por corpus
COSINE_SPACE: str = "cosine"               # métrica de distancia

#  Retrieval 
TOP_K: int = 3               # chunks a recuperar por consulta 
MAX_CHUNKS: int = 500        # límite de chunks en el índice

#  Logging 
LOG_FILE: Path = LOG_DIR / "rag.log"
LOG_LEVEL: str = "INFO"