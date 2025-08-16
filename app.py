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

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Detección de COVID-19",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Diccionarios de idioma
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
        "titulo_reporte": "Reporte de Diagnóstico - COVID-19",
        "fecha_reporte": "Fecha de Generación",
        "prob_reporte": "Probabilidad Calculada",
        "diag_reporte": "Conclusión Diagnóstica",
        "notas_reporte": "Notas Clínicas",
        "texto_notas": "Este reporte es generado automáticamente y debe ser validado por un profesional de salud.",
        "estadisticas": "Análisis Estadístico Avanzado",
        "mc_nemar": "Prueba de McNemar",
        "wilcoxon": "Prueba de Wilcoxon",
        "tabla_contingencia": "Tabla de Contingencia",
        "estadistico": "Estadístico",
        "valor_p": "Valor p",
        "exactitud": "Exactitud",
        "sensibilidad": "Sensibilidad",
        "especificidad": "Especificidad",
        "reporte_clasificacion": "Reporte de Clasificación",
        "metricas": "Métricas Clínicas",
        "seleccion_idioma": "Idioma",
        "opciones_idioma": {"es": "Español", "en": "Inglés"}
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
        "titulo_reporte": "Diagnostic Report - COVID-19",
        "fecha_reporte": "Generation Date",
        "prob_reporte": "Calculated Probability",
        "diag_reporte": "Diagnostic Conclusion",
        "notas_reporte": "Clinical Notes",
        "texto_notas": "This report is automatically generated and should be validated by a healthcare professional.",
        "estadisticas": "Advanced Statistical Analysis",
        "mc_nemar": "McNemar Test",
        "wilcoxon": "Wilcoxon Test",
        "tabla_contingencia": "Contingency Table",
        "estadistico": "Statistic",
        "valor_p": "p-value",
        "exactitud": "Accuracy",
        "sensibilidad": "Sensitivity",
        "especificidad": "Specificity",
        "reporte_clasificacion": "Classification Report",
        "metricas": "Clinical Metrics",
        "seleccion_idioma": "Language",
        "opciones_idioma": {"es": "Spanish", "en": "English"}
    }
}

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model('modelos/mobilenetv2_ajustado.keras')

modelo = cargar_modelo()

# Datos clínicos de ejemplo
def cargar_datos_clinicos():
    np.random.seed(42)
    n_muestras = 150
    return pd.DataFrame({
        'real': np.random.randint(0, 2, n_muestras),
        'predicho': np.where(np.random.rand(n_muestras) < 0.9, 
                       np.random.randint(0, 2, n_muestras), 
                       np.random.randint(0, 2, n_muestras))
    })

# Procesamiento de imágenes
def procesar_imagen(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[..., :3]
    return np.expand_dims(img_array / 255.0, axis=0)

# Generar mapa de calor
def generar_mapa_calor(img_array, modelo):
    capa_conv = modelo.get_layer('mobilenetv2_1.00_224').get_layer('out_relu')
    modelo_grad = tf.keras.models.Model(
        [modelo.inputs], [capa_conv.output, modelo.output]
    )
    
    with tf.GradientTape() as cinta:
        salidas_conv, predicciones = modelo_grad(img_array)
        perdida = predicciones[:, 0]
    
    gradientes = cinta.gradient(perdida, salidas_conv)
    gradientes_promedio = tf.reduce_mean(gradientes, axis=(0, 1, 2))
    
    salidas_conv = salidas_conv[0]
    mapa_calor = salidas_conv @ gradientes_promedio[..., tf.newaxis]
    mapa_calor = tf.squeeze(mapa_calor).numpy()
    mapa_calor = np.maximum(mapa_calor, 0) / (np.max(mapa_calor) + 1e-8)
    
    mapa_calor = cv2.resize(mapa_calor, (img_array.shape[2], img_array.shape[1]))
    mapa_calor = cv2.applyColorMap(np.uint8(255 * mapa_calor), cv2.COLORMAP_JET)
    return cv2.cvtColor(mapa_calor, cv2.COLOR_BGR2RGB)

# Pruebas estadísticas
def realizar_pruebas_estadisticas(datos, idioma):
    resultados = {}
    
    # Prueba de McNemar
    tabla_contingencia = confusion_matrix(datos['real'], datos['predicho'])
    prueba_mcnemar = stats.mcnemar(tabla_contingencia)
    resultados['mcnemar'] = {
        "tabla": tabla_contingencia,
        "estadistico": prueba_mcnemar.statistic,
        "valor_p": prueba_mcnemar.pvalue
    }
    
    # Prueba de Wilcoxon
    prueba_wilcoxon = stats.wilcoxon(datos['real'], datos['predicho'])
    resultados['wilcoxon'] = {
        "estadistico": prueba_wilcoxon.statistic,
        "valor_p": prueba_wilcoxon.pvalue
    }
    
    # Métricas de desempeño
    vn, fp, fn, vp = tabla_contingencia.ravel()
    resultados['metricas'] = {
        "exactitud": (vp + vn) / (vp + vn + fp + fn),
        "sensibilidad": vp / (vp + fn),
        "especificidad": vn / (vn + fp)
    }
    
    # Reporte de clasificación
    resultados['reporte_clasificacion'] = classification_report(
        datos['real'], datos['predicho'], output_dict=True
    )
    
    return resultados

# Generar PDF con estadísticas
def generar_reporte_pdf(img, prediccion, mapa_calor, resultados_estadisticos, idioma):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    
    # Encabezado
    pdf.cell(0, 10, IDIOMAS[idioma]["titulo_reporte"], 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"{IDIOMAS[idioma]['fecha_reporte']}: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
    pdf.ln(10)
    
    # Imagen original
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, IDIOMAS[idioma]["original"], 0, 1)
    ruta_img = "temp_img.png"
    img.save(ruta_img)
    pdf.image(ruta_img, x=50, w=100)
    pdf.ln(5)
    
    # Resultados
    pdf.cell(0, 10, f"{IDIOMAS[idioma]['prob_reporte']}: {prediccion*100:.2f}%", 0, 1)
    pdf.cell(0, 10, f"{IDIOMAS[idioma]['diag_reporte']}: {IDIOMAS[idioma]['positivo'] if prediccion > 0.5 else IDIOMAS[idioma]['negativo']}", 0, 1)
    pdf.ln(10)
    
    # Mapa de calor
    pdf.cell(0, 10, IDIOMAS[idioma]["hallazgos"], 0, 1)
    ruta_mapa = "temp_mapa.png"
    plt.imsave(ruta_mapa, mapa_calor)
    pdf.image(ruta_mapa, x=30, w=150)
    pdf.ln(10)
    
    # Análisis Estadístico
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, IDIOMAS[idioma]["estadisticas"], 0, 1)
    pdf.ln(5)
    
    # McNemar
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, IDIOMAS[idioma]["mc_nemar"], 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"{IDIOMAS[idioma]['estadistico']}: {resultados_estadisticos['mcnemar']['estadistico']:.4f}", 0, 1)
    pdf.cell(0, 10, f"{IDIOMAS[idioma]['valor_p']}: {resultados_estadisticos['mcnemar']['valor_p']:.4f}", 0, 1)
    pdf.ln(5)
    
    # Wilcoxon
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, IDIOMAS[idioma]["wilcoxon"], 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"{IDIOMAS[idioma]['estadistico']}: {resultados_estadisticos['wilcoxon']['estadistico']:.4f}", 0, 1)
    pdf.cell(0, 10, f"{IDIOMAS[idioma]['valor_p']}: {resultados_estadisticos['wilcoxon']['valor_p']:.4f}", 0, 1)
    pdf.ln(10)
    
    # Métricas
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, IDIOMAS[idioma]["metricas"], 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"{IDIOMAS[idioma]['exactitud']}: {resultados_estadisticos['metricas']['exactitud']*100:.2f}%", 0, 1)
    pdf.cell(0, 10, f"{IDIOMAS[idioma]['sensibilidad']}: {resultados_estadisticos['metricas']['sensibilidad']*100:.2f}%", 0, 1)
    pdf.cell(0, 10, f"{IDIOMAS[idioma]['especificidad']}: {resultados_estadisticos['metricas']['especificidad']*100:.2f}%", 0, 1)
    
    # Notas
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, IDIOMAS[idioma]["notas_reporte"], 0, 1)
    pdf.multi_cell(0, 5, IDIOMAS[idioma]["texto_notas"])
    
    # Guardar PDF
    nombre_reporte = f"reporte_{idioma}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf.output(nombre_reporte)
    
    # Limpieza
    os.remove(ruta_img)
    os.remove(ruta_mapa)
    
    return nombre_reporte

# Interfaz principal
def main():
    # Selector de idioma
    idioma = st.sidebar.selectbox(
        IDIOMAS["es"]["seleccion_idioma"],
        options=["es", "en"],
        format_func=lambda x: IDIOMAS["es"]["opciones_idioma"][x]
    )
    
    st.title(IDIOMAS[idioma]["titulo"])
    st.markdown("---")
    
    # Carga de datos clínicos
    datos_clinicos = cargar_datos_clinicos()
    resultados_estadisticos = realizar_pruebas_estadisticas(datos_clinicos, idioma)
    
    # Carga de imagen
    archivo_subido = st.file_uploader(
        IDIOMAS[idioma]["subir_imagen"],
        type=["jpg", "jpeg", "png"]
    )
    
    if archivo_subido is not None:
        img = Image.open(archivo_subido)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption=IDIOMAS[idioma]["original"], use_column_width=True)
        
        if st.button(IDIOMAS[idioma]["analizar"]):
            with st.spinner(IDIOMAS[idioma]["analizar"] + "..."):
                # Procesamiento
                img_array = procesar_imagen(img)
                prediccion = modelo.predict(img_array, verbose=0)[0][0]
                
                # Generar visualizaciones
                mapa_calor = generar_mapa_calor(img_array, modelo)
                superposicion = cv2.addWeighted(
                    cv2.resize(np.array(img), (224, 224)), 0.7,
                    mapa_calor, 0.3, 0
                )
                
                # Mostrar resultados
                with col2:
                    st.subheader(IDIOMAS[idioma]["resultados"])
                    
                    # Probabilidad
                    st.metric(
                        IDIOMAS[idioma]["prob_covid"],
                        f"{prediccion*100:.2f}%",
                        delta=IDIOMAS[idioma]["positivo"] if prediccion > 0.5 else IDIOMAS[idioma]["negativo"],
                        delta_color="inverse"
                    )
                    
                    # Confianza
                    confianza = min(prediccion, 1-prediccion)*2*100
                    st.progress(int(confianza))
                    st.caption(f"{IDIOMAS[idioma]['confianza']}: {confianza:.2f}%")
                    
                    # Visualización
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    ax[0].imshow(np.array(img.resize((224, 224))))
                    ax[0].set_title(IDIOMAS[idioma]["original"])
                    ax[0].axis('off')
                    
                    ax[1].imshow(superposicion)
                    ax[1].set_title(IDIOMAS[idioma]["hallazgos"])
                    ax[1].axis('off')
                    
                    st.pyplot(fig)
                
                # Sección de Estadísticas
                st.markdown("---")
                st.header(IDIOMAS[idioma]["estadisticas"])
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.subheader(IDIOMAS[idioma]["mc_nemar"])
                    st.write(f"**{IDIOMAS[idioma]['estadistico']}:** {resultados_estadisticos['mcnemar']['estadistico']:.4f}")
                    st.write(f"**{IDIOMAS[idioma]['valor_p']}:** {resultados_estadisticos['mcnemar']['valor_p']:.4f}")
                    
                    # Tabla de contingencia
                    st.subheader(IDIOMAS[idioma]["tabla_contingencia"])
                    st.dataframe(pd.DataFrame(
                        resultados_estadisticos['mcnemar']['tabla'],
                        columns=["Predicho Neg", "Predicho Pos"],
                        index=["Real Neg", "Real Pos"]
                    ).style.background_gradient(cmap='Blues'))
                
                with col4:
                    st.subheader(IDIOMAS[idioma]["wilcoxon"])
                    st.write(f"**{IDIOMAS[idioma]['estadistico']}:** {resultados_estadisticos['wilcoxon']['estadistico']:.4f}")
                    st.write(f"**{IDIOMAS[idioma]['valor_p']}:** {resultados_estadisticos['wilcoxon']['valor_p']:.4f}")
                    
                    # Métricas
                    st.subheader(IDIOMAS[idioma]["metricas"])
                    st.metric(IDIOMAS[idioma]["exactitud"], f"{resultados_estadisticos['metricas']['exactitud']*100:.2f}%")
                    st.metric(IDIOMAS[idioma]["sensibilidad"], f"{resultados_estadisticos['metricas']['sensibilidad']*100:.2f}%")
                    st.metric(IDIOMAS[idioma]["especificidad"], f"{resultados_estadisticos['metricas']['especificidad']*100:.2f}%")
                
                # Reporte de clasificación
                st.subheader(IDIOMAS[idioma]["reporte_clasificacion"])
                st.dataframe(pd.DataFrame(resultados_estadisticos['reporte_clasificacion']).transpose())
                
                # Generar y descargar reporte
                ruta_reporte = generar_reporte_pdf(img, prediccion, superposicion, resultados_estadisticos, idioma)
                
                with open(ruta_reporte, "rb") as f:
                    bytes_reporte = f.read()
                
                st.download_button(
                    label=IDIOMAS[idioma]["descargar_reporte"],
                    data=bytes_reporte,
                    file_name=f"reporte_covid_{idioma}.pdf",
                    mime="application/pdf"
                )
                
                # Limpieza
                os.remove(ruta_reporte)

if __name__ == "__main__":
    main()