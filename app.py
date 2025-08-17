# -*- coding: utf-8 -*-
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import os
from datetime import datetime
from fpdf import FPDF
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report

# ======================
# CONFIGURACIÓN INICIAL
# ======================
st.set_page_config(
    page_title="Sistema de Detección de COVID-19",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# MULTILENGUAJE
# ======================
IDIOMAS = {
    "es": {
        "titulo": "Sistema de Detección de COVID-19",
        "subir_imagen": "Subir radiografía de tórax",
        "analizar": "Analizar Imagen",
        "resultados": "Resultados del Análisis",
        "prob_covid": "Probabilidad de COVID-19",
        "diagnostico": "Diagnóstico",
        "positivo": "POSITIVO para SARS-CoV-2",
        "negativo": "NEGATIVO para SARS-CoV-2",
        "confianza": "Confianza del Modelo",
        "descargar_reporte": "Descargar Reporte",
        "hallazgos": "Hallazgos Radiológicos",
        "original": "Imagen Original",
        "mapa_calor": "Mapa de Calor",
        "estadisticas": "Estadísticas Clínicas",
        "mc_nemar": "Prueba de McNemar",
        "wilcoxon": "Prueba de Wilcoxon",
        "metricas": "Métricas de Rendimiento"
    },
    "en": {
        "titulo": "COVID-19 Detection System",
        "subir_imagen": "Upload chest X-ray",
        "analizar": "Analyze Image",
        "resultados": "Analysis Results",
        "prob_covid": "COVID-19 Probability",
        "diagnostico": "Diagnosis",
        "positivo": "SARS-CoV-2 POSITIVE",
        "negativo": "SARS-CoV-2 NEGATIVE",
        "confianza": "Model Confidence",
        "descargar_reporte": "Download Report",
        "hallazgos": "Radiological Findings",
        "original": "Original Image",
        "mapa_calor": "Heatmap",
        "estadisticas": "Clinical Statistics",
        "mc_nemar": "McNemar Test",
        "wilcoxon": "Wilcoxon Test",
        "metricas": "Performance Metrics"
    }
}

# ======================
# CARGAR MODELO
# ======================
@st.cache_resource
def cargar_modelo():
    try:
        modelo = tf.keras.models.load_model('models/mobilenetv2_finetuned.keras')
        # Prueba de inferencia para verificar carga correcta
        dummy_input = np.zeros((1, 224, 224, 3))
        modelo.predict(dummy_input, verbose=0)
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

# ======================
# FUNCIONES DE PROCESAMIENTO
# ======================
def procesar_imagen(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    return img_array / 255.0

def generar_mapa_calor(img_array, modelo):
    capa_conv = modelo.get_layer('mobilenetv2_1.00_224').get_layer('out_relu')
    grad_model = tf.keras.models.Model(
        [modelo.inputs], [capa_conv.output, modelo.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(np.expand_dims(img_array, 0))
        loss = preds[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = np.maximum(tf.squeeze(heatmap).numpy(), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# ======================
# GENERAR REPORTE PDF
# ======================
def generar_reporte_pdf(imagen, prediccion, mapa_calor, idioma):
    pdf = FPDF()
    pdf.add_page()
    
    # Encabezado
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, IDIOMAS[idioma]["titulo"], 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1, 'C')
    
    # Imagen y resultados
    temp_img = "temp_img.png"
    imagen.save(temp_img)
    pdf.image(temp_img, x=50, w=100)
    pdf.ln(10)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Probabilidad COVID-19: {prediccion*100:.2f}%", 0, 1)
    pdf.cell(0, 10, f"Diagnóstico: {IDIOMAS[idioma]['positivo'] if prediccion > 0.5 else IDIOMAS[idioma]['negativo']}", 0, 1)
    
    # Mapa de calor
    temp_heat = "temp_heat.png"
    plt.imsave(temp_heat, mapa_calor)
    pdf.image(temp_heat, x=30, w=150)
    
    # Limpieza
    os.remove(temp_img)
    os.remove(temp_heat)
    
    # Guardar PDF
    pdf_path = f"reporte_covid_{datetime.now().strftime('%Y%m%d%H%M')}.pdf"
    pdf.output(pdf_path)
    return pdf_path

# ======================
# INTERFAZ PRINCIPAL
# ======================
def main():
    # Selector de idioma
    idioma = st.sidebar.selectbox("Idioma", ["es", "en"])
    
    st.title(IDIOMAS[idioma]["titulo"])
    
    # Carga de modelo
    modelo = cargar_modelo()
    
    # Carga de imagen
    img_file = st.file_uploader(IDIOMAS[idioma]["subir_imagen"], type=["jpg", "jpeg", "png"])
    
    if img_file and modelo:
        img = Image.open(img_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption=IDIOMAS[idioma]["original"], use_column_width=True)
        
        if st.button(IDIOMAS[idioma]["analizar"]):
            with st.spinner("Analizando..."):
                # Procesamiento
                img_array = procesar_imagen(img)
                prediccion = modelo.predict(np.expand_dims(img_array, 0), verbose=0)[0][0]
                
                # Generar visualizaciones
                heatmap = generar_mapa_calor(img_array, modelo)
                overlay = cv2.addWeighted(
                    cv2.resize(np.array(img_array), (224, 224)), 0.7,
                    heatmap, 0.3, 0
                )
                
                # Mostrar resultados
                with col2:
                    st.subheader(IDIOMAS[idioma]["resultados"])
                    
                    # Probabilidad y diagnóstico
                    prob = prediccion * 100
                    st.metric(
                        IDIOMAS[idioma]["prob_covid"],
                        f"{prob:.2f}%",
                        delta=IDIOMAS[idioma]["positivo"] if prediccion > 0.5 else IDIOMAS[idioma]["negativo"],
                        delta_color="inverse"
                    )
                    
                    # Visualización
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    ax[0].imshow(img_array)
                    ax[0].set_title(IDIOMAS[idioma]["original"])
                    ax[0].axis('off')
                    
                    ax[1].imshow(overlay)
                    ax[1].set_title(IDIOMAS[idioma]["hallazgos"])
                    ax[1].axis('off')
                    st.pyplot(fig)
                
                # Generar reporte PDF
                pdf_path = generar_reporte_pdf(img, prediccion, overlay, idioma)
                
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                
                st.download_button(
                    label=IDIOMAS[idioma]["descargar_reporte"],
                    data=pdf_bytes,
                    file_name=f"reporte_covid_{idioma}.pdf",
                    mime="application/pdf"
                )
                
                os.remove(pdf_path)

if __name__ == "__main__":
    main()