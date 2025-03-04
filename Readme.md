# UNIR - Máster Universitario en Inteligencia Artificial - TFM
## Estudio piloto de chatbot de asesoría fiscal para la liquidación del IVA en España
Presentado por: Jaime González Rodríguez / Director: Guillermo Torralba Elipe / Fecha: 05/03/2025
Este repositorio contiene el código fuente del chatbot y los modelos BETO y mT5 entrenados para que el chatbot sea capaz de responder preguntas fiscales.
Se han otorgado permisos a Streamlit para tomar este repositorio como fuente para construir la aplicación web de demo. En resumen, en este repositorio se puede encontrar:
- El código fuente de la aplicación web descrito en la memoria del TFM:
  - Scripts de python utilizados por streamlit community cloud ("streamlit_app.py" e "inferencia.py").
  - Los modelos entrenados BETO (1-ClasificadorPreguntas) y mT5 (2-GeneradorRespuesta).
  - El fichero de base de conocimiento (BaseConocimiento).
- Los conjuntos de datos utilizados para entrenar los modelos y la base de conocimiento (DatasetsEntrenamiento).
- Los jupyter notebooks utilizados para entrenar los modelos  (JupyterNotebooks). Estos archivos han sido exportados desde Google Colab Pro.

