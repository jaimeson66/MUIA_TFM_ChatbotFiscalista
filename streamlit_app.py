import streamlit as st
import logging
from PIL import Image, ImageEnhance
import time
import torch
import json
import requests
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from inferencia import generar_respuesta

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
NUMBER_OF_MESSAGES_TO_DISPLAY = 20
API_DOCS_URL = "https://docs.streamlit.io/library/api-reference"

# Streamlit Page Configuration
st.set_page_config(
    page_title="Streamly - Asistente virtual de IVA",
    layout="wide"
)

# Streamlit Title
st.title("Asistente fiscal IVA")

def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def long_running_task(duration):
    """
    Simula una operación de larga duración

    Parameters:
    - duration: int, duration of the task in seconds

    Returns:
    - str: Completion message
    """
    time.sleep(duration)
    return "Long-running operation completed."

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)


def initialize_conversation():
    """
    Inicia el historial de la conversación con el sistema. Incluye
    los mensajes del usuario y el sistema.

    Returns:
    - list: Initialized conversation history.
    """
    assistant_message = "¡Hola! Soy su asistente de consultas fiscales ¿En qué puedo ayudarle?"

    conversation_history = [
        {"role": "system", "content": "Inicio conversacion"},
        {"role": "assistant", "content": assistant_message}
    ]
    return conversation_history

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
#-------------ESTA FUNCION ES LA IMPORTANTE)
def on_chat_submit(chat_input):
    """
    Gestiona la interacción del usuario cuando introduce el mensaje

    Parametros:
    - chat_input (str): The chat input from the user.
    - latest_updates (dict): The latest Streamlit updates fetched from a JSON file or API.

    Salida:
    - Ninguna: Se actualiza el historial del chat
    """
    user_input = chat_input.strip().lower()

    #if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = initialize_conversation()

    st.session_state.conversation_history.append({"role": "user", "content": user_input})

    assistant_reply = generar_respuesta(user_input)


    st.session_state.conversation_history.append({"role": "assistant", "content": assistant_reply})
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": assistant_reply})


def initialize_session_state():
    """Initialize session state variables."""
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

    # Insert custom CSS for glowing effect
    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow:
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 45px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
