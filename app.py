# -*- coding: utf-8 -*-
"""
Aplicación Web de Despliegue en Streamlit (Diseño Clínico Premium)
Predicción de Riesgo de Ataque al Corazón (Contexto Médico)
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Configuración de página
st.set_page_config(
    page_title="Predicción de Riesgo de Ataque al Corazón",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inyección de estilos CSS para replicar el diseño de la imagen
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Fondo general con degradado azul oscuro profundo */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0d1e44 0%, #060e22 75%, #040917 100%) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #f1f5f9 !important;
    }

    /* Ocultar barra superior y footer por defecto de Streamlit */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1200px !important;
    }

    /* Cabecera Principal */
    .header-container {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    .header-icon-circle {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 68px;
        height: 68px;
        background: linear-gradient(135deg, #ff5757 0%, #ff6b6b 100%);
        border-radius: 50%;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(255, 87, 87, 0.35);
    }
    .header-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        line-height: 1.25;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        max-width: 680px;
        margin: 0.8rem auto 0 auto;
        line-height: 1.5;
    }

    /* Tarjetas principales estilo Glassmorphism */
    .glass-card {
        background: rgba(11, 26, 56, 0.75);
        border: 1px solid rgba(34, 57, 107, 0.7);
        border-radius: 20px;
        padding: 1.8rem 2rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
        margin-bottom: 1.5rem;
    }

    .card-title-row {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 1.5rem;
    }
    .badge-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        background: rgba(30, 58, 115, 0.6);
        border: 1px solid rgba(56, 95, 179, 0.5);
        border-radius: 10px;
        color: #60a5fa;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }

    /* Subtarjetas de campos de formulario */
    .field-card {
        background: rgba(8, 19, 44, 0.65);
        border: 1px solid rgba(27, 48, 92, 0.6);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease;
    }
    .field-card:hover {
        border-color: rgba(59, 130, 246, 0.5);
    }
    .field-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.4rem;
        color: #cbd5e1;
        font-size: 0.88rem;
        font-weight: 600;
    }

    /* Estilos personalizados para inputs de Streamlit */
    div[data-testid="stSlider"] {
        padding-top: 0.2rem;
    }
    div[data-testid="stSlider"] label {
        display: none !important;
    }
    div[data-testid="stSelectbox"] label {
        display: none !important;
    }
    
    /* Personalización del Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #08142e !important;
        border-color: #1a325c !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
    }
    div[data-baseweb="select"] svg {
        fill: #94a3b8 !important;
    }

    /* Personalización del Slider (Color Coral) */
    div[data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg, #ff5757, #ff6b6b) !important;
    }
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #ff5757 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 10px rgba(255, 87, 87, 0.5) !important;
    }

    /* Botón de acción principal (Coral con flecha) */
    div.stButton > button {
        background: linear-gradient(90deg, #ff5252 0%, #ff6b6b 100%) !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.5rem !important;
        width: 100% !important;
        box-shadow: 0 8px 20px rgba(255, 82, 82, 0.35) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        margin-top: 0.5rem;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 28px rgba(255, 82, 82, 0.5) !important;
    }
    div.stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Tarjeta lateral de Evaluación Clínica */
    .eval-card {
        background: rgba(11, 26, 56, 0.75);
        border: 1px solid rgba(34, 57, 107, 0.7);
        border-radius: 20px;
        padding: 1.8rem 1.8rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
        min-height: 480px;
        display: flex;
        flex-direction: column;
    }
    .eval-badge-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        background: rgba(255, 87, 87, 0.12);
        border: 1px solid rgba(255, 87, 87, 0.35);
        border-radius: 12px;
        color: #ff5757;
        margin-bottom: 1rem;
    }
    .eval-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.8rem 0;
    }
    .eval-prompt {
        font-size: 0.9rem;
        color: #94a3b8;
        line-height: 1.5;
    }
    .ecg-wave {
        margin-top: auto;
        opacity: 0.25;
        width: 100%;
    }

    /* Tarjeta de Nota Metodológica inferior */
    .method-card {
        background: rgba(8, 20, 48, 0.7);
        border: 1px solid rgba(26, 48, 96, 0.6);
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        margin-top: 1rem;
    }
    .info-icon-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
        color: #60a5fa;
        flex-shrink: 0;
    }
    .method-text-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #60a5fa;
        margin: 0 0 0.2rem 0;
    }
    .method-text-desc {
        font-size: 0.82rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.4;
    }

    /* Resultados de la Predicción */
    .result-box-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.25) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 14px;
        padding: 1.2rem;
        margin-top: 1rem;
        text-align: center;
    }
    .result-box-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-radius: 14px;
        padding: 1.2rem;
        margin-top: 1rem;
        text-align: center;
    }
    .result-status-title {
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .status-high { color: #f87171; }
    .status-low { color: #34d399; }
    .prob-metric {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CARGA DEL MODELO
# ==============================================================================
filename = 'modelo-class.pkl'
if not os.path.exists(filename):
    alt = r"C:\Users\danie\Downloads\modelo-class.pkl"
    if os.path.exists(alt):
        filename = alt

modelo, labelencoder, variables, min_max_scaler = None, None, None, None
if os.path.exists(filename):
    with open(filename, 'rb') as f:
        modelo, labelencoder, variables, min_max_scaler = pickle.load(f)
else:
    st.error(f"No se encontró el archivo '{filename}'. Asegúrate de que esté en el mismo directorio.")

# ==============================================================================
# ENCABEZADO PRINCIPAL (Exacto a la imagen)
# ==============================================================================
st.markdown("""
<div class="header-container">
    <div class="header-icon-circle">
        <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
            <path d="M3.22 12H9.5l1.5-3 2 6.5 1.5-3.5h6.28"/>
        </svg>
    </div>
    <h1 class="header-title">Predicción de Riesgo de Ataque al Corazón</h1>
    <p class="header-subtitle">
        Esta aplicación utiliza un modelo de Machine Learning (SVM) con ponderación balanceada 
        entrenado con datos clínicos para estimar el riesgo de que un paciente sufra un ataque al corazón 
        (ACV o evento cardiovascular).
    </p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# DISPOSICIÓN EN 2 COLUMNAS (Formulario | Evaluación Clínica)
# ==============================================================================
col_form, col_eval = st.columns([2.15, 1], gap="large")

with col_form:
    # Contenedor Tarjeta: Datos Clínicos del Paciente
    st.markdown("""
    <div class="card-title-row">
        <div class="badge-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect width="16" height="20" x="4" y="2" rx="2"/>
                <line x1="8" x2="16" y1="6" y2="6"/>
                <line x1="12" x2="12" y1="10" y2="16"/>
                <line x1="9" x2="15" y1="13" y2="13"/>
            </svg>
        </div>
        <h2 class="card-title">Datos Clínicos del Paciente</h2>
    </div>
    """, unsafe_allow_html=True)

    # Grid 2x3 de Campos Clínicos
    c1, c2 = st.columns(2)

    with c1:
        # 1. Edad
        st.markdown("""
        <div class="field-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
            <span>Edad (años)</span>
        </div>
        """, unsafe_allow_html=True)
        age = st.slider("edad_slider", min_value=1, max_value=82, value=50, step=1, key="slider_age")

        # 2. Hipertensión
        st.markdown("""
        <div class="field-header" style="margin-top: 0.8rem;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>
            <span>¿Padece hipertensión?</span>
        </div>
        """, unsafe_allow_html=True)
        hypertension = st.selectbox("hipertension_select", ['No', 'Yes'], key="select_hyp")

        # 3. Enfermedad Cardíaca
        st.markdown("""
        <div class="field-header" style="margin-top: 0.8rem;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            <span>¿Padece enfermedad cardíaca previa?</span>
        </div>
        """, unsafe_allow_html=True)
        heart_disease = st.selectbox("heart_select", ['No', 'Yes'], key="select_hd")

    with c2:
        # 4. Estado Civil
        st.markdown("""
        <div class="field-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><path d="m15 7-3 3-3-3"/><path d="m15 17-3-3-3 3"/></svg>
            <span>¿Ha estado casado alguna vez?</span>
        </div>
        """, unsafe_allow_html=True)
        ever_married = st.selectbox("married_select", ['No', 'Yes'], key="select_marr")

        # 5. Glucosa promedio
        st.markdown("""
        <div class="field-header" style="margin-top: 0.8rem;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>
            <span>Glucosa promedio (mg/dL)</span>
        </div>
        """, unsafe_allow_html=True)
        avg_glucose_level = st.slider("glucosa_slider", min_value=55, max_value=272, value=105, step=1, key="slider_gluc")

        # 6. Estado de fumador
        st.markdown("""
        <div class="field-header" style="margin-top: 0.8rem;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><path d="M18 8c0-2.5-2-2.5-2-5"/><path d="M22 8c0-2.5-2-2.5-2-5"/><rect width="18" height="8" x="2" y="12" rx="2"/></svg>
            <span>Estado de fumador</span>
        </div>
        """, unsafe_allow_html=True)
        smoking_options = ["never smoked", "formerly smoked", "smokes", "Unknown"]
        smoking_status_user = st.selectbox("smoke_select", smoking_options, key="select_smoke")
        # En el dataset original algunas categorías llevan comillas simples literales
        smoking_status = f"'{smoking_status_user}'" if smoking_status_user in ["never smoked", "formerly smoked"] else smoking_status_user

    # Botón Principal
    btn_evaluar = st.button("❤️  Evaluar riesgo clínico  →", use_container_width=True)

    # Tarjeta de Nota Metodológica (Inferior izquierda)
    st.markdown("""
    <div class="method-card">
        <div class="info-icon-badge">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
        </div>
        <div>
            <h4 class="method-text-title">Nota Metodológica:</h4>
            <p class="method-text-desc">
                El modelo de predicción utilizado es una Máquina de Vectores de Soporte (SVM) con ponderación balanceada de clases.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


with col_eval:
    # Tarjeta Lateral Derecha: Evaluación Clínica
    st.markdown("""
    <div class="eval-card">
        <div class="eval-badge-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="M9.5 12h1.5l1.5-3 2 6 1-3h2.5"/>
            </svg>
        </div>
        <h3 class="eval-title">Evaluación clínica</h3>
    """, unsafe_allow_html=True)

    if not btn_evaluar:
        st.markdown("""
        <p class="eval-prompt">
            Completa los datos del paciente y presiona el botón para estimar el riesgo.
        </p>
        <svg class="ecg-wave" viewBox="0 0 500 120" fill="none" stroke="#3b82f6" stroke-width="2">
            <path d="M0,60 L120,60 L140,20 L160,100 L180,40 L200,80 L220,60 L500,60"/>
        </svg>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Ejecución del Modelo al presionar el botón
        if modelo is not None and variables is not None and min_max_scaler is not None:
            # 1. Construir DataFrame
            df_input = pd.DataFrame([[age, hypertension, heart_disease, ever_married, float(avg_glucose_level), smoking_status]],
                                    columns=['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'smoking_status'])
            
            # 2. Preprocesamiento idéntico al entrenamiento
            df_proc = pd.get_dummies(df_input, columns=['smoking_status', 'hypertension', 'heart_disease', 'ever_married'], drop_first=False, dtype=int)
            df_proc = df_proc.reindex(columns=variables, fill_value=0)
            df_proc[['age', 'avg_glucose_level']] = min_max_scaler.transform(df_proc[['age', 'avg_glucose_level']])

            # 3. Inferencia
            pred_num = modelo.predict(df_proc)
            pred_label = labelencoder.inverse_transform(pred_num)[0]

            prob = None
            if hasattr(modelo, "predict_proba"):
                prob = modelo.predict_proba(df_proc)[0, 1] * 100

            # 4. Renderizar resultado estilizado
            if pred_label == 'Yes':
                prob_str = f"{prob:.1f}%" if prob is not None else "Elevada"
                st.markdown(f"""
                <div class="result-box-high">
                    <div class="result-status-title status-high">⚠️ ALTO RIESGO</div>
                    <div class="prob-metric">{prob_str}</div>
                    <p style="font-size: 0.8rem; color: #fca5a5; margin: 0.4rem 0 0 0;">
                        Probabilidad estimada de evento cardiovascular
                    </p>
                </div>
                <div style="margin-top: 1.2rem; font-size: 0.84rem; color: #cbd5e1; line-height: 1.5;">
                    <strong style="color: #f87171;">Recomendación Clínica:</strong><br>
                    Se sugiere ordenar de forma prioritaria electrocardiograma, perfil lipídico y valoración médica preventiva.
                </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                prob_str = f"{prob:.1f}%" if prob is not None else "Baja"
                st.markdown(f"""
                <div class="result-box-low">
                    <div class="result-status-title status-low">✅ BAJO RIESGO</div>
                    <div class="prob-metric">{prob_str}</div>
                    <p style="font-size: 0.8rem; color: #6ee7b7; margin: 0.4rem 0 0 0;">
                        Probabilidad estimada de evento cardiovascular
                    </p>
                </div>
                <div style="margin-top: 1.2rem; font-size: 0.84rem; color: #cbd5e1; line-height: 1.5;">
                    <strong style="color: #34d399;">Recomendación Clínica:</strong><br>
                    El paciente no presenta indicadores críticos. Mantener hábitos saludables y chequeos de rutina preventiva.
                </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Error al cargar el modelo.")
            st.markdown("</div>", unsafe_allow_html=True)