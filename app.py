from __future__ import annotations

import random
import time
from collections.abc import Iterator

import streamlit as st

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

# Respuesta simulada hasta que responder() esté lista
RESPUESTAS = [
    "Puedo ayudarte con información sobre tarifas e instalaciones deportivas de Madrid.",
    "Consulta sobre abonos, piscinas, polideportivos y normativa municipal.",
    "Buena pregunta. Déjame buscar en el corpus de Deporte Municipal Madrid.",
]

# Sidebar
with st.sidebar:
    st.header("Configuración")
    top_k = st.slider("TOP_K (chunks recuperados)", min_value=1, max_value=10, value=3)
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

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    respuesta = f"{random.choice(RESPUESTAS)}\n\n_(Simulado — TOP_K: {top_k})_"

    with st.chat_message("assistant"):
        escrito = st.write_stream(stream_palabras(respuesta))
        contenido = escrito if isinstance(escrito, str) else respuesta

    st.session_state.messages.append({"role": "assistant", "content": contenido})