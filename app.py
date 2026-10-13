# -*- coding: utf-8 -*-
"""
Aplicación Web de Despliegue en Streamlit
Predicción de Riesgo de Ataque al Corazón (Contexto Médico)
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Configuración de página
st.set_page_config(
    page_title="Predicción de Riesgo Cardiovascular",
    page_icon="❤️",
    layout="centered"
)

# Título y encabezado clínico
st.title("❤️ Predicción de Riesgo de Ataque al Corazón")
st.markdown("""
Esta aplicación utiliza un modelo de **Machine Learning (SVM con ponderación balanceada)** 
entrenado con datos clínicos para estimar el riesgo de que un paciente sufra un ataque al corazón / ACV (*stroke*).
""")

# Carga del modelo
filename = 'modelo-class.pkl'
if not os.path.exists(filename):
    st.error(f"❌ Error: No se encontró el archivo '{filename}'. Asegúrate de colocarlo en la misma carpeta que 'app.py'.")
    st.stop()

with open(filename, 'rb') as f:
    modelo, labelencoder, variables, min_max_scaler = pickle.load(f)

# Formulario de entrada de datos
st.subheader("📋 Datos Clínicos del Paciente")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Edad (años)", min_value=1, max_value=82, value=50, step=1)
    hypertension = st.selectbox("¿Padece Hipertensión?", ["No", "Yes"])
    heart_disease = st.selectbox("¿Padece Enfermedad Cardíaca Previa?", ["No", "Yes"])

with col2:
    ever_married = st.selectbox("¿Ha estado casado alguna vez?", ["No", "Yes"])
    avg_glucose_level = st.slider("Nivel Promedio de Glucosa en Sangre (mg/dL)", min_value=55.0, max_value=272.0, value=105.0, step=0.5)
    smoking_status = st.selectbox(
        "Estado de Fumador",
        ["'never smoked'", "'formerly smoked'", "smokes", "Unknown"],
        help="Seleccione el historial de tabaquismo del paciente."
    )

# Construcción del DataFrame con los datos ingresados
datos = [[age, hypertension, heart_disease, ever_married, avg_glucose_level, smoking_status]]
data = pd.DataFrame(datos, columns=['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'smoking_status'])

# Botón para ejecutar la predicción
st.markdown("---")
if st.button("🔍 Evaluar Riesgo Clínico", type="primary", use_container_width=True):
    # 1. Preparación de datos (Dummies con drop_first=False)
    data_preparada = data.copy()
    data_preparada['avg_glucose_level'] = pd.to_numeric(data_preparada['avg_glucose_level'], errors='coerce')
    data_preparada = pd.get_dummies(data_preparada, columns=['smoking_status', 'hypertension', 'heart_disease', 'ever_married'], drop_first=False, dtype=int)

    # 2. Alineación de columnas con el orden de entrenamiento
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

    # 3. Normalización con el scaler ya calibrado
    variables_numericas = ['age', 'avg_glucose_level']
    data_preparada[variables_numericas] = min_max_scaler.transform(data_preparada[variables_numericas])

    # 4. Predicción con SVM
    Y_pred_num = modelo.predict(data_preparada)
    Y_pred_etiqueta = labelencoder.inverse_transform(Y_pred_num)[0]

    # Probabilidad de ataque (clase 'Yes')
    if hasattr(modelo, "predict_proba"):
        probabilidad = modelo.predict_proba(data_preparada)[0, 1] * 100
    else:
        probabilidad = None

    # 5. Visualización del Resultado
    st.subheader("📊 Resultado del Diagnóstico Preliminar")

    if Y_pred_etiqueta == 'Yes':
        st.error(f"""
        ### ⚠️ ALERTA MÉDICA: ALTO RIESGO DE ATAQUE AL CORAZÓN
        El modelo ha clasificado al paciente como **caso de riesgo positivo (Yes)**.
        {f'**Probabilidad estimada de riesgo:** {probabilidad:.1f}%' if probabilidad else ''}
        
        **Recomendación Clínica:**
        Se sugiere ordenar inmediatamente exámenes cardiovasculares complementarios (electrocardiograma, perfil lipídico) 
        y evaluación médica prioritaria.
        """)
    else:
        st.success(f"""
        ### ✅ RESULTADO: BAJO RIESGO PRELIMINAR
        El modelo no detecta indicadores críticos de ataque al corazón en este momento (**No**).
        {f'**Probabilidad estimada de riesgo:** {probabilidad:.1f}%' if probabilidad else ''}
        
        **Recomendación Clínica:**
        Mantener controles preventivos de rutina y hábitos de vida saludables.
        """)

    # Mostrar la tabla de datos procesados
    data['Prediccion'] = Y_pred_etiqueta
    if probabilidad is not None:
        data['Probabilidad_Riesgo_%'] = np.round(probabilidad, 2)
    st.dataframe(data, use_container_width=True)

# Nota metodológica al pie
st.markdown("---")
st.info("""
**Nota Metodológica:**  
El modelo desplegado es una **Máquina de Vectores de Soporte (SVM)** con ponderación balanceada (`class_weight='balanced'`). 
Fue seleccionado específicamente por optimizar la **Sensibilidad / Recall (84%)**, minimizando así los **Falsos Negativos** 
(evitando catalogar a pacientes en riesgo como sanos), lo cual es el factor más crítico en medicina.
""")