import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration,MT5ForConditionalGeneration,AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
import numpy as np

# Cargar el modelo y el tokenizer
ruta_modelo_clasificador = "./1-ClasificadorPreguntas"
model_clasificador = BertForSequenceClassification.from_pretrained(ruta_modelo_clasificador)
tokenizer_clasificador = BertTokenizer.from_pretrained(ruta_modelo_clasificador)


def clasificador_pregunta(frase_entrada):
   # Tokenizar el texto
   inputs_class = tokenizer_clasificador(
      input_sentence,
      return_tensors="pt",  # Formato PyTorch
      padding="max_length",
      truncation=False,
      max_length=512
  )

   # Mover los tensores a la GPU, si estuviera disponible
   device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
   model_clasificador.to(device)
   inputs_class = {key: val.to(device) for key, val in inputs_class.items()}

   # Inferencia
   with torch.no_grad():
       outputs = model_clasificador(**inputs_class)
       logits = outputs.logits
       inferencia_clase = torch.argmax(logits, dim=-1).item()

   return inferencia_clase


def generar_respuesta(pregunta):
    ruta_modelo_generativo = "./2-GeneradorRespuesta"
    tokenizer_generativo = T5Tokenizer.from_pretrained(ruta_modelo_generativo)
    model_generativo = T5ForConditionalGeneration.from_pretrained(ruta_modelo_generativo)
    # Importar datos desde base de conocimiento
    pd.set_option("display.max_colwidth", None)  #Esto es para no truncar la columna de respuesta
    df_etiquetas = pd.read_csv('./BaseConocimiento/baseConocimiento.csv',encoding = 'utf-8', delimiter = ';', index_col=False)
    dato = df_etiquetas[df_etiquetas["Clase"] == clasificador_pregunta(pregunta)]
    contexto = dato["Contexto"].to_string(index=False)
    
    #
    input_text = f"context: {contexto} question: {pregunta}"
    inputs = tokenizer_generativo(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Mover los tensores a la GPU, si estuviera disponible
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_generativo.to(device)  # Move the generative model to the device

    # Mover los tensores a la GPU, si estuviera disponible
    inputs = {key: val.to(device) for key, val in inputs.items()}

    outputs = model_generativo.generate(**inputs, max_length=128, num_beams=3, early_stopping=True)
    return tokenizer_generativo.decode(outputs[0], skip_special_tokens=True)
