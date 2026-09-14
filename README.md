# project-rag-hmd-deporte-municipal

La interfaz web permite interactuar con el sistema RAG de forma visual.

## Paso 0 — Instalación y arranque

```bash
git clone https://github.com/dcruzpuri/project-rag-hmd-deporte-municipal.git
cd project-rag-hmd-deporte-municipal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Paso 1 — Ejecutar la app

```bash
streamlit run app.py
```

### Funcionalidades
- Chat con historial de conversación
- Contexto y chunks recuperados visibles
- Métricas por consulta (TOP_K, nº chunks, tiempo, modelo)
- Selector de TOP_K en el sidebar