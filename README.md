# RAG Deporte Municipal Madrid

Asistente conversacional que responde preguntas sobre el deporte municipal de Madrid usando Retrieval-Augmented Generation (RAG). Los datos vienen del portal de datos abiertos del Ayuntamiento de Madrid.

## Equipo

- **Héctor** — corpus, pipeline de indexación, ChromaDB
- **Miguel** — retrieval, generación, orquestación (`responder()`)
- **David** — interfaz Streamlit, evaluación, documentación

## Requisitos

- Python 3.10+
- Ollama instalado y corriendo (`ollama serve`)
- API key de Google Gemini

## Instalación

```bash
git clone https://github.com/dcruzpuri/project-rag-hmd-deporte-municipal.git
cd project-rag-hmd-deporte-municipal
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración

Copia el archivo de ejemplo y rellena tus claves:

```bash
cp .env.example .env
```

Variables necesarias en `.env`:

EMBED_PROVIDER="ollama"
OLLAMA_EMBED_MODEL="qwen3-embedding:0.6b"
GEN_PROVIDER="google"
GOOGLE_API_KEY=tu_clave_aqui
GEMINI_API_KEY=tu_clave_aqui



## Indexación

Antes de usar el asistente hay que generar el índice vectorial:

```bash
ollama serve  # en una terminal aparte
python main.py --index
```

Con un Mac M5 tarda unos 16 minutos. Con GPU NVIDIA tarda 5-8 minutos.

## Uso

### Interfaz web

```bash
streamlit run app.py
```

### CLI

```bash
python main.py --ask "¿Cuánto cuesta el abono de piscina?"
python main.py --query "piscinas municipales"
```

## Estructura

project-rag-hmd-deporte-municipal/
├── app.py # Interfaz Streamlit
├── main.py # CLI
├── config.py # Configuración global
├── src/
│ ├── load.py # Carga de documentos
│ ├── chunk.py # Chunking
│ ├── embed.py # Embeddings
│ ├── index.py # Indexación en ChromaDB
│ ├── retrieve.py # Retrieval semántico
│ ├── prompts.py # Construcción de prompts
│ ├── generate.py # Generación con Gemini
│ └── logic.py # Orquestación (responder())
├── data/ # Corpus (CSVs y PDFs)
├── queries/ # Preguntas de evaluación
├── entregables/ # Informe de decisiones
└── output/ # ChromaDB (no se sube a Git)


## Evaluación

Las preguntas de evaluación están en `queries/evaluation_questions.json`. Son 22 preguntas: 18 sobre el corpus y 4 fuera de dominio para probar la abstención. Las hemos intercalado para evitar detección de patrones.
