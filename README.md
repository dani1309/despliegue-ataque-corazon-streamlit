# Predicción de Riesgo de Ataque al Corazón (Streamlit App)

Aplicación web interactiva desarrollada con **Streamlit** para estimar el riesgo clínico de que un paciente sufra un ataque al corazón / ACV (*stroke*).

## Modelo Utilizado
- **Algoritmo:** Support Vector Machine (SVM) con kernel RBF y ponderación balanceada (class_weight='balanced').
- **Enfoque Médico:** Optimizado para maximizar el **Recall / Sensibilidad (84%)**, reduciendo al mínimo los Falsos Negativos (pacientes en riesgo que no reciben atención médica temprana).

## Archivos del Repositorio
- \pp.py\: Código principal de la interfaz web en Streamlit.
- \modelo-class.pkl\: Artefacto serializado con el modelo entrenado, escalador (\MinMaxScaler\), codificador (\LabelEncoder\) y nombres de variables.
- \equirements.txt\: Dependencias necesarias para ejecutar la aplicación en Streamlit Cloud.

## Despliegue en Streamlit Cloud
1. Conecta tu cuenta de GitHub en [share.streamlit.io](https://share.streamlit.io).
2. Selecciona este repositorio (\dani1309/despliegue-ataque-corazon-streamlit\).
3. Selecciona \pp.py\ como el archivo principal.
4. Haz clic en **Deploy**.
