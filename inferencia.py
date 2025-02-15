import torch
from transformers import MT5ForConditionalGeneration,AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
import torch.nn.functional as F  # Para usar softmax
import sentencepiece
import pandas as pd
import torch
import numpy as np
import unicodedata
import os

# Depurar texto de entrada (para BETO). Se reutiliza función usada
# para procesar los datos de entrada en el entrenamiento.

def limpiar_texto(texto):
    if isinstance(texto, str):
        # Eliminar tildes
        texto = ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))
        # Eliminar signos de interrogación sustituyéndolos por espacios en blanco
        texto = texto.replace('¿', '').replace('?', '').replace(',','').replace('-','').replace('_','')
        # Convertir a minúsculas
        texto = texto.lower()
    return texto


def clasificador_pregunta(input_sentence, umbral_confianza=0.25):
    """
    Clasifica una pregunta y devuelve la clase predicha.
    Si la confianza no está dentro del umbral, devuelve "clase no reconocida".

    :param input_sentence: Frase de entrada a clasificar
    :param umbral_confianza: Valor mínimo de confianza para aceptar la clasificación
    :return: Clase predicha o "clase no reconocida"
    """
    # 1) Cargar el modelo y el tokenizer
    ruta_modelo_clasificador = "./1-ClasificadorPreguntas"
    model_clasificador = BertForSequenceClassification.from_pretrained(ruta_modelo_clasificador)
    tokenizer_clasificador = AutoTokenizer.from_pretrained(ruta_modelo_clasificador)
    # Tokenizar el texto
    inputs_class = tokenizer_clasificador(
        limpiar_texto(input_sentence), # Cadena de texto limpia
        return_tensors="pt",  # Formato PyTorch
        padding="max_length",
        truncation=False,
        max_length=512
    )

    # Seleccionar el dispositivo para cargar el modelo. En streamlit no está disponible GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_clasificador.to(device)
    # Introducir los inputs en el dispositivo
    inputs_class = {key: val.to(device) for key, val in inputs_class.items()}

    # Inferencia
    with torch.no_grad():
        outputs = model_clasificador(**inputs_class)
        logits = outputs.logits
        probabilidades = F.softmax(logits, dim=-1)  # Obtener probabilidades

        # Imprimir para depuración
        #print(f"Logits: {logits}")
        #print(f"Probabilidades: {probabilidades}")

        # Softmax para probabilidad
        confianza_maxima, inferencia_clase = torch.max(probabilidades, dim=-1)  # Confianza y clase predicha

    # Verificar si la confianza está dentro del umbral
    if confianza_maxima.item() < umbral_confianza:
        #print(f"Confianza baja ({confianza_maxima.item():.2f}), devolviendo respuesta genérica.")
        return "clase no reconocida"
    else:
        #print(f"Predicted class: {inferencia_clase.item()}, Confianza: {confianza_maxima.item():.2f}")
        return inferencia_clase.item()


def generar_respuesta(pregunta):
    """
    Generación de la respuesta a una pregunta.

    :param pregunta: Pregunta introducida por el usuario.
    :return: Respuesta (si la pregunta está correctamente clasificada)
             o mensaje de error (si la pregunta no está correctamente clasificada).
    """
    # 1) Importar el modelo generativo fine tuneado y tokenizar según MT5.
    ruta_modelo_generativo = "./2-GeneradorRespuesta"
    tokenizer_generativo = AutoTokenizer.from_pretrained.from_pretrained(ruta_modelo_generativo)
    model_generativo = MT5ForConditionalGeneration.from_pretrained(ruta_modelo_generativo)

    # 2)Importar datos desde base de conocimiento
    pd.set_option("display.max_colwidth", None)  #Esto es para no truncar la columna de respuesta
    df_etiquetas = pd.read_csv('./BaseConocimiento/baseConocimiento.csv',encoding = 'utf-8', delimiter = ';', index_col=False)

    # 3) La generación de texto solamente debe hacerse si se ha reconocido la pregunta
    if clasificador_pregunta(pregunta) != "clase no reconocida":
      # 4) Consultar el contexto según la clase identificada
      dato = df_etiquetas[df_etiquetas["Clase"] == clasificador_pregunta(pregunta)]
      contexto = dato["Contexto"].to_string(index=False)
      """
      Contexto: procede de la base de conocimiento
      pregunta: es el input del usuario
      """

      input_text = f"context: {contexto} question: {pregunta}"
      inputs = tokenizer_generativo(input_text, return_tensors="pt", max_length=512, truncation=True)

      # 5) Seleccionar GPU como dispositivo si estuviera disponible. En streamlit es CPU
      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
      model_generativo.to(device)

      # 6) Mover inputs al dispositivo.
      for key in inputs:
          inputs[key] = inputs[key].to(model_generativo.device)

      outputs = model_generativo.generate(**inputs, max_length=256,
                                          num_beams=2,
                                          no_repeat_ngram_size=2,
                                          early_stopping=False,
                                          temperature=0.8,
                                          top_p=0.95,
                                          repetition_penalty=1.2,
                                          do_sample=True
                                          )
      return tokenizer_generativo.decode(outputs[0], skip_special_tokens=True)
    else:
      return "No comprendo la pregunta. Por favor ¿Puede volver a repetirla?"
