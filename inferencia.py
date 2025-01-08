import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration,MT5ForConditionalGeneration,AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import BertTokenizer, BertForSequenceClassification
import sentencepiece
import pandas as pd
import torch
import numpy as np

# Cargar el modelo y el tokenizer
ruta_modelo_clasificador = "./1-ClasificadorPreguntas"
model_clasificador = BertForSequenceClassification.from_pretrained(ruta_modelo_clasificador)
tokenizer_clasificador = BertTokenizer.from_pretrained(ruta_modelo_clasificador)


def clasificador_pregunta(input_sentence, umbral_confianza=0.25):
    """
    Clasifica una pregunta y devuelve la clase predicha.
    Si la confianza no está dentro del umbral, devuelve "clase no reconocida".

    :param input_sentence: Frase de entrada a clasificar
    :param umbral_confianza: Valor mínimo de confianza para aceptar la clasificación
    :return: Clase predicha o "clase no reconocida"
    """
    # Tokenizar el texto
    inputs_class = tokenizer_clasificador(
        input_sentence,
        return_tensors="pt",  # Formato PyTorch
        padding="max_length",
        truncation=False,
        max_length=512
    )

    # Mover los tensores al dispositivo adecuado
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_clasificador.to(device)
    inputs_class = {key: val.to(device) for key, val in inputs_class.items()}

    # Inferencia
    with torch.no_grad():
        outputs = model_clasificador(**inputs_class)
        logits = outputs.logits
        probabilidades = F.softmax(logits, dim=-1)  # Obtener probabilidades

        # Imprimir para depuración
        #print(f"Logits: {logits}")
        #print(f"Probabilidades: {probabilidades}")

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
    ruta_modelo_generativo = "./2-GeneradorRespuesta"
    tokenizer_generativo = T5Tokenizer.from_pretrained(ruta_modelo_generativo)
    model_generativo = T5ForConditionalGeneration.from_pretrained(ruta_modelo_generativo)
    # Importar datos desde base de conocimiento
    pd.set_option("display.max_colwidth", None)  #Esto es para no truncar la columna de respuesta
    df_etiquetas = pd.read_csv('./BaseConocimiento/baseConocimiento.csv',encoding = 'utf-8', delimiter = ';', index_col=False)
    if clasificador_pregunta(pregunta) != "clase no reconocida":
      dato = df_etiquetas[df_etiquetas["Clase"] == clasificador_pregunta(pregunta)]
      contexto = dato["Contexto"].to_string(index=False)

      #
      input_text = f"context: {contexto} question: {pregunta}"
      inputs = tokenizer_generativo(input_text, return_tensors="pt", max_length=512, truncation=True)

      # Mover inputs a la GPU, si estuviera disponible
      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
      model_generativo.to(device)  # Move the generative model to the device

      # Mover inputs a la GPU, si estuviera disponible
      inputs = {key: val.to(device) for key, val in inputs.items()}

      outputs = model_generativo.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
      return tokenizer_generativo.decode(outputs[0], skip_special_tokens=True)
    else:
      return "No comprendo la pregunta. Por favor ¿Puede volver a repetirla?"
