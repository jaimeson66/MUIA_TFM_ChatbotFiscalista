import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration,MT5ForConditionalGeneration,AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
import numpy as np

# Cargar el modelo y el tokenizer
modelo_clasificacion_dir = "./1-ClasificadorPreguntas"
model = BertForSequenceClassification.from_pretrained(modelo_clasificacion_dir)
tokenizer = BertTokenizer.from_pretrained(modelo_clasificacion_dir)


def clasificador_pregunta(frase_entrada):
   # Tokenizar el texto
   inputs = tokenizer(
      frase_entrada,
      return_tensors="pt",  # Formato PyTorch
      padding="max_length",
      truncation=False,
      max_length=512
   )

   # Mover los tensores al dispositivo adecuado
   device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
   model.to(device)
   inputs = {key: val.to(device) for key, val in inputs.items()}

   # Inferencia
   with torch.no_grad():
      outputs = model(**inputs)
      logits = outputs.logits
      inferencia_clase = torch.argmax(logits, dim=-1).item()

   return inferencia_clase


def generar_respuesta(pregunta):
    model_name = "timpal0l/mdeberta-v3-base-squad2"  # Modelo ya afinado en SQuAD en español
    model_SQUAD = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer_SQUAD = AutoTokenizer.from_pretrained(model_name)
    # Importar datos desde base de conocimiento
    pd.set_option("display.max_colwidth", None)  #Esto es para no truncar la columna de respuesta
    df_etiquetas = pd.read_csv('./BaseConocimiento/baseConocimiento.csv',encoding = 'utf-8', delimiter = ';', index_col=False)
    dato = df_etiquetas[df_etiquetas["Clase"] == clasificador_pregunta(pregunta)]
    contexto = dato["Respuesta"].to_string(index=False)
    respuesta = contexto
    # Formatear la pregunta y el contexto en español
    #pregunta = "cuando se presenta una declaracion rectificativa"
    #pregunta = "En que modelo se informa una operacion intracomunitaria"
    #prediccion = predictClass(pregunta)
    #dato = df_etiquetas[df_etiquetas["Clase"] == prediccion]
    #contexto =  dato["Respuesta"].to_string(index=False)
    #contexto = "las operaciones intracomunitarias deben informarse en el modelo 349"

    inputs = tokenizer_SQUAD(
        pregunta,
        contexto,
        return_tensors="pt",  # Tensores de PyTorch
        truncation=True,      # Truncar si el texto es demasiado largo
        max_length=512       # Longitud máxima admitida por el modelo
    )
    outputs = model_SQUAD(**inputs)

    # Extraer las posiciones de inicio y fin con mayor probabilidad
    start_idx = outputs.start_logits.argmax()
    end_idx = outputs.end_logits.argmax()



    respuesta = tokenizer_SQUAD.convert_tokens_to_string(
        tokenizer_SQUAD.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx + 1])
    )

    return respuesta
