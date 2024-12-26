import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration,MT5ForConditionalGeneration,AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def generar_respuesta(pregunta):
    model_name = "timpal0l/mdeberta-v3-base-squad2"  # Modelo ya afinado en SQuAD en español
    model_SQUAD = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer_SQUAD = AutoTokenizer.from_pretrained(model_name)
    # Formatear la pregunta y el contexto en español
    #pregunta = "cuando se presenta una declaracion rectificativa"
    #pregunta = "En que modelo se informa una operacion intracomunitaria"
    #prediccion = predictClass(pregunta)
    #dato = df_etiquetas[df_etiquetas["Clase"] == prediccion]
    #contexto =  dato["Respuesta"].to_string(index=False)
    contexto = "las operaciones intracomunitarias deben informarse en el modelo 349"

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
