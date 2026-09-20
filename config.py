"""
config.py
Todas las constantes del proyecto. Se leen de variables de entorno (.env)
con valores por defecto: se cambia de proveedor sin tocar el código.

Hay 3 interruptores INDEPENDIENTES (cada uno con su proveedor y su modelo):
  - EMBED: embeddings (fase offline + consulta)
  - GEN:   generación (fase online, futuro generate.py)
  - TAG:   etiquetado (fase offline; por defecto sigue a GEN)
Cada proveedor (ollama / huggingface / google) tiene su variable de modelo
(OLLAMA_* / HF_* / GOOGLE_*); EMBED_MODEL, GEN_MODEL y TAG_MODEL se
resuelven automáticamente según el proveedor de cada interruptor.
"""
import os

# Carga .env si existe (python-dotenv es opcional: si no está instalado, se ignora)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Interruptor 1: EMBEDDINGS (offline + consulta)
EMBED_PROVIDER: str = os.getenv("EMBED_PROVIDER", "ollama")   # "ollama" | "huggingface" | "google"

OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_EMBED_MODEL: str = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GOOGLE_EMBED_MODEL: str = os.getenv("GOOGLE_EMBED_MODEL", "gemini-embedding-2")

# Interruptor 2: GENERACIÓN (fase online)
GEN_PROVIDER: str = os.getenv("GEN_PROVIDER", "ollama")      # "ollama" | "huggingface" | "google"
OLLAMA_GEN_MODEL: str = os.getenv("OLLAMA_GEN_MODEL", "llama3.2")  # llama3.2, llama4:scout (ollama) / gemini-2.0-flash (google)
HF_GEN_MODEL: str = os.getenv("HF_GEN_MODEL", "mistralai/Mistral-7B-v0.1")
GOOGLE_GEN_MODEL: str = os.getenv("GOOGLE_GEN_MODEL", "gemini-2.0-flash")


# Interruptor 3: TAGGING (offline; por defecto sigue a GEN)
# Si en .env se deja vacío, hereda el proveedor de GEN.
TAG_PROVIDER: str = os.getenv("TAG_PROVIDER") or GEN_PROVIDER
# Modelos de tagging por proveedor: si en .env se dejan vacíos, heredan los de GEN.
# El patrón `os.getenv(...) or <parámetro>` hace que un valor vacío también herede.
OLLAMA_TAG_MODEL: str = os.getenv("OLLAMA_TAG_MODEL") or OLLAMA_GEN_MODEL
HF_TAG_MODEL: str = os.getenv("HF_TAG_MODEL") or HF_GEN_MODEL
GOOGLE_TAG_MODEL: str = os.getenv("GOOGLE_TAG_MODEL") or GOOGLE_GEN_MODEL


def _resolver(provider: str, ollama_model: str, hf_model: str, google_model: str, switch: str) -> str:
    """Devuelve el modelo que usa cada interruptor según su proveedor."""
    modelo = {"ollama": ollama_model, "huggingface": hf_model, "google": google_model}.get(provider)
    if modelo is None:
        raise ValueError(f"{switch} no soportado: {provider!r}")
    return modelo


# Modelo efectivo de cada interruptor (el que usan embed.py, tag.py, futuro generate.py)
EMBED_MODEL: str = _resolver(EMBED_PROVIDER, OLLAMA_EMBED_MODEL, HF_EMBED_MODEL, GOOGLE_EMBED_MODEL, "EMBED_PROVIDER")
GEN_MODEL: str = _resolver(GEN_PROVIDER, OLLAMA_GEN_MODEL, HF_GEN_MODEL, GOOGLE_GEN_MODEL, "GEN_PROVIDER")
TAG_MODEL: str = _resolver(TAG_PROVIDER, OLLAMA_TAG_MODEL, HF_TAG_MODEL, GOOGLE_TAG_MODEL, "TAG_PROVIDER")

# --- Ollama (offline, compartido por los 3 interruptores) ---
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# --- HuggingFace (online, compartido) ---
HF_DEVICE: str = os.getenv("HF_DEVICE", "cpu")   # "cpu" | "cuda"
HF_TOKEN: str | None = os.getenv("HF_TOKEN") or None  # solo modelos gated (aquellos que requieren aceptar la licencia y un token de acceso)

# --- Google (online, compartido) ---
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY") or None

# --- Chunking ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))


# --- Retrieval (online) ---
TOP_K: int = int(os.getenv("TOP_K", "5"))
MAX_CHUNKS: int = int(os.getenv("MAX_CHUNKS", "5"))

# --- Generación (online) ---
GEN_TEMPERATURE: float = float(os.getenv("GEN_TEMPERATURE", "0.2"))
ABSTENTION_MESSAGE: str = os.getenv(
    "ABSTENTION_MESSAGE",
    "No dispongo de esa información en los documentos proporcionados.",
)


# --- Embeddings ---
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "384"))  # 384 (all-MiniLM) / 3072 (gemini)
EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "30"))
# Lectura de cada POST a Ollama (/api/embed): en carga fría con el modelo TAG
# residente, el primer batch puede tardar mucho; default 600 s cubre el offload.
EMBED_TIMEOUT: int = int(os.getenv("EMBED_TIMEOUT", "600"))
EXPORT_EMBEDDINGS: bool = os.getenv("EXPORT_EMBEDDINGS", "true").lower() == "true"
# Caché offline del preflight de disponibilidad (src/embed.py): máxima
# dimensión que maneja el modelo cuando la comprobación online no se puede
# hacer (red cortada / sin API key). Vacío = no declarado (dim modelo = None).
#   EMBED_DIM_MAX_OLLAMA / EMBED_DIM_MAX_HF / EMBED_DIM_MAX_GOOGLE
def _dim_max(proveedor: str) -> int | None:
    abrev = {"ollama": "OLLAMA", "huggingface": "HF", "google": "GOOGLE"}[proveedor]
    raw = (os.getenv(f"EMBED_DIM_MAX_{abrev}") or "").strip()
    return int(raw) if raw else None


EMBED_DIM_MAX_OLLAMA: int | None = _dim_max("ollama")
EMBED_DIM_MAX_HF: int | None = _dim_max("huggingface")
EMBED_DIM_MAX_GOOGLE: int | None = _dim_max("google")

# --- TSD: tagging + scoring + dedup ---
TAG_SCORING_DEDUP: bool = os.getenv("TAG_SCORING_DEDUP", "true").lower() == "true"
DEDUP_UMBRAL: float = float(os.getenv("DEDUP_UMBRAL", "0.93"))  # punto de partida para all-minilm-l6-v2
# Auditoría de pares descartados: una línea JSON por descarte (coseno/clave,
# similitud del par, chunk afectado y vencedor) para inspeccionar por qué y
# contra qué se descartó cada chunk. Vacío = sin auditoría.
DEDUP_AUDIT: bool = os.getenv("DEDUP_AUDIT", "true").lower() == "true"
DEDUP_AUDIT_RUTA: str = os.getenv("DEDUP_AUDIT_RUTA", "output/dedup_audit.jsonl")

# --- ChromaDB ---
CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./output/chroma_db")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "deporte_municipal")
COSINE_SPACE: str = "cosine"