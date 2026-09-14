"""
app.py — Asistente RAG Deporte Municipal Madrid
"""

from __future__ import annotations
import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="RAG Deporte Municipal Madrid",
    page_icon="soccer",
    layout="centered",
)

def mensaje_bienvenida():
    """
    Muestra un mensaje de bienvenida en la aplicación.
    """
    st.title("Bienvenido al Asistente RAG Deporte Municipal Madrid")
    st.write(
        "¡Hola! Soy tu **asistente virtual** para ayudarte con información sobre el Deporte Municipal de Madrid. ¿En qué puedo ayudarte?"
    )

#Empezar conversación
if "messages" not in st.session_state:
    st.session_state.messages = [mensaje_bienvenida()]

#Título de la aplicación
st.title("Asistente RAG Deporte Municipal Madrid")
st.caption("Consulta información sobre el Deporte Municipal de Madrid de manera rápida y sencilla.")

