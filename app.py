from __future__ import annotations

import time
from collections.abc import Iterator

import streamlit as st
from src.logic import responder

# Configuración de página
st.set_page_config(
    page_title="RAG Deporte Municipal Madrid",
    page_icon="⚽",
    layout="centered",
)

# Streaming simulado
def stream_palabras(texto: str, delay: float = 0.04) -> Iterator[str]:
    for palabra in texto.split():
        yield palabra + " "
        time.sleep(delay)

# Mensaje de bienvenida
def mensaje_bienvenida() -> dict:
    return {
        "role": "assistant",
        "content": (
            "¡Hola! Soy tu **asistente virtual** para ayudarte con información "
            "sobre el Deporte Municipal de Madrid. ¿En qué puedo ayudarte?"
        ),
    }


# Sidebar
with st.sidebar:
    st.header("Configuración")
    top_k = st.slider("TOP_K (chunks recuperados)", min_value=1, max_value=5, value=3)
    st.divider()
    st.markdown("Corpus disponible")
    st.markdown("- Tarifas instalaciones (PDF)")
    st.markdown("- Polideportivos Madrid (CSV)")
    st.markdown("- Piscinas municipales (PDF)")
    st.markdown("- Reglamento instalaciones (PDF)")
    st.divider()
    if st.button("Limpiar chat", use_container_width=True):
        st.session_state.messages = [mensaje_bienvenida()]
        st.rerun()

# Título de la aplicación
st.title("🏟️ Asistente RAG — Deporte Municipal Madrid")
st.caption("Consulta información sobre instalaciones, tarifas y normativa deportiva de Madrid.")

# Inicializar historial
if "messages" not in st.session_state:
    st.session_state.messages = [mensaje_bienvenida()]

if "last_resultado" not in st.session_state:
    st.session_state.last_resultado = None

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Buscando en el corpus..."):
        resultado = responder(prompt, top_k=top_k)

    with st.chat_message("assistant"):
        if resultado.get("error"):
            st.error(resultado["error"])
        else:
            escrito = st.write_stream(stream_palabras(resultado["respuesta"]))
            contenido = escrito if isinstance(escrito, str) else resultado["respuesta"]
            st.session_state.messages.append({"role": "assistant", "content": contenido})
    st.session_state.last_resultado = resultado
    
# Mostrar fuentes, contexto y métricas del último resultado
if st.session_state.last_resultado:
    resultado = st.session_state.last_resultado
    metrics = resultado.get("metrics", {})
    fuentes = resultado.get("fuentes", [])
    chunks = resultado.get("chunks", [])

    # Fuentes
    if fuentes:
        with st.expander("Fuentes utilizadas"):
            for fuente in fuentes:
                st.markdown(f"- `{fuente}`")

    # Contexto recuperado
    with st.expander("Contexto recuperado (debug)"):
        if chunks:
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}** — `{chunk.get('source', 'desconocido')}`")
                st.text(chunk.get("text", ""))
                st.divider()
        else:
            st.text(resultado.get("contexto", "Sin contexto"))

    # Métricas
    st.markdown("### Métricas")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("TOP_K", metrics.get("top_k", top_k))
    col2.metric("Chunks", metrics.get("n_chunks", "-"))
    col3.metric("Modelo", metrics.get("model", "-"))
    col4.metric("Abstención", "Sí" if resultado.get("abstained") else "No")