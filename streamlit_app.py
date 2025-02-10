import streamlit as st
import logging
from PIL import Image, ImageEnhance
import time
import torch
import json
import requests
from inferencia import generar_respuesta

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Constantes
NUMBER_OF_MESSAGES_TO_DISPLAY = 20

# Configuración de la página de streamlit
st.set_page_config(
    page_title="Streamly - Asistente virtual de IVA",
    layout="wide"
)

# Título de la página
st.title("Asistente fiscal IVA")

@st.cache_data(show_spinner=False)

def initialize_conversation():
    """
    Inicia el historial de la conversación con el sistema. Incluye
    los mensajes del usuario y los generados automáticamente.
    Salida:
    - lista: Conversación inicializada con lista vacía para añadir los mensajes que se van incorporando
    """
    assistant_message = "¡Hola! Soy su asistente de consultas fiscales ¿En qué puedo ayudarle?"
    conversation_history = [
        {"role": "system", "content": "Inicio conversacion"},
        {"role": "assistant", "content": assistant_message}
    ]
    return conversation_history

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
#-------------ESTA FUNCION ES FUNDAMENTAL-----------------
def on_chat_submit(chat_input):
    """
    Gestiona la interacción del usuario cuando introduce el mensaje
    Parametros:
    - chat_input (str): El texto enviado por el usuario
    Salida:
    - Ninguna. Se actualiza el historial del chat
    """
    user_input = chat_input.strip().lower()

    #Si no hay conversación en la sesión se muestra el mensaje de bienvenida
    st.session_state.conversation_history = initialize_conversation()
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    assistant_reply = generar_respuesta(user_input)
    st.session_state.conversation_history.append({"role": "assistant", "content": assistant_reply})
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": assistant_reply})


def initialize_session_state():
    """Iniciar variables de estado."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

def main():
    """
    Función principal que mantiene el chat en funcionamiento
    """
    initialize_session_state()

    if not st.session_state.history:
        initial_bot_message = "¡Hola! Soy su asistente de consultas fiscales ¿En qué puedo ayudarle?"
        st.session_state.history.append({"role": "assistant", "content": initial_bot_message})
        st.session_state.conversation_history = initialize_conversation()

    chat_input = st.chat_input("Pregúnteme")
    if chat_input:
        on_chat_submit(chat_input)

    # Mostrar historial de chat
    for message in st.session_state.history[-NUMBER_OF_MESSAGES_TO_DISPLAY:]:
        role = message["role"]
        avatar_image = "imgs/avatar_streamly.png" if role == "assistant" else "imgs/stuser.png" if role == "user" else None
        with st.chat_message(role):
            st.write(message["content"])

if __name__ == "__main__":
    main()
