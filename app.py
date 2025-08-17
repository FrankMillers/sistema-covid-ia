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
import io
from datetime import datetime
from fpdf import FPDF
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import requests
import tempfile

# ======================
# CONFIGURACIÓN INICIAL
# ======================
st.set_page_config(
    page_title="Sistema de Detección de COVID-19",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# ESTILOS CSS PERSONALIZADOS
# ======================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .positive-result {
        background: #ffebee;
        border-left: 5px solid #f44336;
    }
    .negative-result {
        background: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# MULTILENGUAJE
# ======================
IDIOMAS = {
    "es": {
        "titulo": "🫁 Sistema de Detección de COVID-19",
        "subtitulo": "Análisis de Radiografías de Tórax con IA",
        "subir_imagen": "📋 Subir radiografía de tórax",
        "formato_info": "Formatos soportados: JPG, JPEG, PNG",
        "analizar": "🔍 Analizar Imagen",
        "resultados": "📊 Resultados del Análisis",
        "prob_covid": "Probabilidad de COVID-19",
        "diagnostico": "Diagnóstico",
        "positivo": "🔴 POSITIVO para SARS-CoV-2",
        "negativo": "🟢 NEGATIVO para SARS-CoV-2",
        "confianza": "Confianza del Modelo",
        "descargar_reporte": "📄 Descargar Reporte PDF",
        "hallazgos": "🔍 Hallazgos Radiológicos",
        "original": "Imagen Original",
        "mapa_calor": "Mapa de Calor (Grad-CAM)",
        "estadisticas": "📈 Estadísticas Clínicas",
        "interpretacion": "💡 Interpretación de Resultados",
        "disclaimer": "⚠️ Aviso Médico",
        "disclaimer_text": "Este sistema es una herramienta de apoyo diagnóstico. Los resultados deben ser interpretados por un profesional médico calificado.",
        "cargando_modelo": "🔄 Cargando modelo de IA...",
        "procesando": "🔄 Procesando imagen...",
        "error_carga": "❌ Error al cargar la imagen. Verifica el formato.",
        "modelo_cargado": "✅ Modelo cargado correctamente",
        "modelo_error": "❌ Error al cargar el modelo"
    },
    "en": {
        "titulo": "🫁 COVID-19 Detection System",
        "subtitulo": "Chest X-ray Analysis with AI",
        "subir_imagen": "📋 Upload chest X-ray",
        "formato_info": "Supported formats: JPG, JPEG, PNG",
        "analizar": "🔍 Analyze Image",
        "resultados": "📊 Analysis Results",
        "prob_covid": "COVID-19 Probability",
        "diagnostico": "Diagnosis",
        "positivo": "🔴 SARS-CoV-2 POSITIVE",
        "negativo": "🟢 SARS-CoV-2 NEGATIVE",
        "confianza": "Model Confidence",
        "descargar_reporte": "📄 Download PDF Report",
        "hallazgos": "🔍 Radiological Findings",
        "original": "Original Image",
        "mapa_calor": "Heatmap (Grad-CAM)",
        "estadisticas": "📈 Clinical Statistics",
        "interpretacion": "💡 Results Interpretation",
        "disclaimer": "⚠️ Medical Disclaimer",
        "disclaimer_text": "This system is a diagnostic support tool. Results should be interpreted by a qualified medical professional.",
        "cargando_modelo": "🔄 Loading AI model...",
        "procesando": "🔄 Processing image...",
        "error_carga": "❌ Error loading image. Check format.",
        "modelo_cargado": "✅ Model loaded successfully",
        "modelo_error": "❌ Error loading model"
    }
}

# ======================
# CONFIGURACIÓN DEL MODELO
# ======================
MODEL_URL = "https://drive.google.com/uc?export=download&id=TU_ID_DEL_MODELO"  # Reemplaza con tu URL
MODEL_PATH = "models/mobilenetv2_finetuned.keras"

# ======================
# FUNCIONES DE UTILIDAD
# ======================
@st.cache_data
def descargar_modelo_desde_url(url, path):
    """Descarga el modelo desde una URL si no existe localmente"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if not os.path.exists(path):
            with st.spinner("Descargando modelo..."):
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Error descargando modelo: {str(e)}")
        return False

@st.cache_resource
def cargar_modelo():
    """Carga el modelo de TensorFlow"""
    try:
        # Intentar cargar desde ruta local primero
        if os.path.exists(MODEL_PATH):
            modelo = tf.keras.models.load_model(MODEL_PATH)
        else:
            # Si no existe, intentar descargar (opcional)
            st.warning("Modelo no encontrado en la ruta local. Asegúrate de que el archivo existe.")
            return None
        
        # Verificar que el modelo funciona
        dummy_input = np.zeros((1, 224, 224, 3))
        _ = modelo.predict(dummy_input, verbose=0)
        
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

def validar_imagen(img):
    """Valida que la imagen sea correcta"""
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Verificar dimensiones mínimas
        if img.size[0] < 50 or img.size[1] < 50:
            return False, "Imagen demasiado pequeña"
        
        return True, img
    except Exception as e:
        return False, str(e)

def procesar_imagen(img):
    """Procesa la imagen para el modelo"""
    try:
        # Redimensionar a 224x224
        img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convertir a array numpy
        img_array = np.array(img_resized)
        
        # Asegurar que tenga 3 canales
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[-1] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Normalizar (0-1)
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    except Exception as e:
        st.error(f"Error procesando imagen: {str(e)}")
        return None

def generar_mapa_calor(img_array, modelo):
    """Genera mapa de calor usando Grad-CAM"""
    try:
        # Encontrar la última capa convolucional
        last_conv_layer = None
        for layer in reversed(modelo.layers):
            if len(layer.output_shape) == 4:  # Capa convolucional
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            # Usar una capa específica conocida de MobileNetV2
            try:
                last_conv_layer = modelo.get_layer('global_average_pooling2d')
                # Si no funciona, buscar por patrón
                for layer in modelo.layers:
                    if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
                        last_conv_layer = layer
            except:
                return np.zeros((224, 224, 3))
        
        # Crear modelo para gradientes
        grad_model = tf.keras.models.Model(
            inputs=[modelo.inputs],
            outputs=[last_conv_layer.output, modelo.output]
        )
        
        # Calcular gradientes
        with tf.GradientTape() as tape:
            inputs = tf.cast(np.expand_dims(img_array, 0), tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            loss = predictions[:, 0]  # COVID-19 probability
        
        # Obtener gradientes
        grads = tape.gradient(loss, conv_outputs)
        
        # Pooling de gradientes
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiplicar feature maps por gradientes
        conv_outputs = conv_outputs[0]
        for i in range(len(pooled_grads)):
            conv_outputs = conv_outputs[:, :, i] * pooled_grads[i]
        
        # Promedio y normalización
        heatmap = tf.reduce_mean(conv_outputs, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Redimensionar al tamaño original
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Aplicar colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    except Exception as e:
        st.warning(f"No se pudo generar mapa de calor: {str(e)}")
        return np.zeros((224, 224, 3))

def crear_overlay(img_array, heatmap, alpha=0.6):
    """Crea overlay de imagen original con mapa de calor"""
    try:
        img_uint8 = (img_array * 255).astype(np.uint8)
        overlay = cv2.addWeighted(img_uint8, alpha, heatmap, 1-alpha, 0)
        return overlay
    except:
        return img_array

def interpretar_resultado(probabilidad, idioma):
    """Interpreta el resultado del modelo"""
    if probabilidad > 0.8:
        if idioma == "es":
            return "🔴 Alta probabilidad de COVID-19", "Se observan patrones consistentes con infección por SARS-CoV-2"
        else:
            return "🔴 High COVID-19 probability", "Patterns consistent with SARS-CoV-2 infection observed"
    elif probabilidad > 0.6:
        if idioma == "es":
            return "🟡 Probabilidad moderada de COVID-19", "Se sugiere evaluación médica adicional"
        else:
            return "🟡 Moderate COVID-19 probability", "Additional medical evaluation suggested"
    elif probabilidad > 0.4:
        if idioma == "es":
            return "🟡 Resultado incierto", "Se requiere análisis médico profesional"
        else:
            return "🟡 Uncertain result", "Professional medical analysis required"
    else:
        if idioma == "es":
            return "🟢 Baja probabilidad de COVID-19", "No se observan patrones típicos de COVID-19"
        else:
            return "🟢 Low COVID-19 probability", "No typical COVID-19 patterns observed"

def generar_reporte_pdf(imagen, prediccion, heatmap, overlay, idioma):
    """Genera reporte PDF mejorado"""
    pdf = FPDF()
    pdf.add_page()
    
    # Configurar fuente para soportar caracteres especiales
    pdf.set_font('Arial', 'B', 16)
    
    # Encabezado
    pdf.cell(0, 10, IDIOMAS[idioma]["titulo"].replace("🫁 ", ""), 0, 1, 'C')
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", 0, 1, 'C')
    pdf.cell(0, 10, f"Sistema de IA - MobileNetV2", 0, 1, 'C')
    pdf.ln(10)
    
    # Resultados principales
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "RESULTADOS DEL ANALISIS", 0, 1, 'L')
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Probabilidad COVID-19: {prediccion*100:.2f}%", 0, 1)
    
    diagnostico = IDIOMAS[idioma]['positivo'] if prediccion > 0.5 else IDIOMAS[idioma]['negativo']
    pdf.cell(0, 10, f"Diagnostico: {diagnostico.replace('🔴 ', '').replace('🟢 ', '')}", 0, 1)
    
    interpretacion, descripcion = interpretar_resultado(prediccion, idioma)
    pdf.cell(0, 10, f"Interpretacion: {interpretacion.replace('🔴 ', '').replace('🟡 ', '').replace('🟢 ', '')}", 0, 1)
    pdf.ln(10)
    
    # Disclaimer
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "AVISO MEDICO IMPORTANTE", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, IDIOMAS[idioma]["disclaimer_text"])
    
    # Guardar temporalmente
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(temp_pdf.name)
    
    return temp_pdf.name

# ======================
# INTERFAZ PRINCIPAL
# ======================
def main():
    # Configuración de idioma
    col_lang, col_space = st.columns([1, 4])
    with col_lang:
        idioma = st.selectbox("🌐 Idioma / Language", ["es", "en"], label_visibility="collapsed")
    
    # Encabezado principal
    st.markdown(f"""
    <div class="main-header">
        <h1>{IDIOMAS[idioma]["titulo"]}</h1>
        <p>{IDIOMAS[idioma]["subtitulo"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar con información
    with st.sidebar:
        st.markdown("### ℹ️ Información del Sistema")
        st.info("""
        **Modelo**: MobileNetV2 Fine-tuned
        **Precisión**: ~94%
        **Datos de entrenamiento**: >10,000 radiografías
        **Tiempo de análisis**: <5 segundos
        """)
        
        st.markdown("### 🎯 Cómo usar")
        st.markdown("""
        1. Sube una radiografía de tórax
        2. Haz clic en 'Analizar'
        3. Revisa los resultados
        4. Descarga el reporte PDF
        """)
        
        # Disclaimer
        st.markdown(f"### {IDIOMAS[idioma]['disclaimer']}")
        st.warning(IDIOMAS[idioma]["disclaimer_text"])
    
    # Cargar modelo
    with st.spinner(IDIOMAS[idioma]["cargando_modelo"]):
        modelo = cargar_modelo()
    
    if modelo is None:
        st.error(IDIOMAS[idioma]["modelo_error"])
        st.stop()
    else:
        st.success(IDIOMAS[idioma]["modelo_cargado"])
    
    # Subida de imagen
    st.markdown(f"## {IDIOMAS[idioma]['subir_imagen']}")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        img_file = st.file_uploader(
            IDIOMAS[idioma]["formato_info"],
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
    
    with col2:
        analizar_btn = st.button(
            IDIOMAS[idioma]["analizar"],
            disabled=(img_file is None),
            use_container_width=True
        )
    
    if img_file is not None:
        try:
            # Cargar y validar imagen
            img = Image.open(img_file)
            is_valid, resultado = validar_imagen(img)
            
            if not is_valid:
                st.error(f"{IDIOMAS[idioma]['error_carga']}: {resultado}")
                return
            
            img = resultado
            
            # Mostrar imagen original
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {IDIOMAS[idioma]['original']}")
                st.image(img, use_column_width=True)
            
            # Análisis
            if analizar_btn:
                with col2:
                    with st.spinner(IDIOMAS[idioma]["procesando"]):
                        # Procesar imagen
                        img_array = procesar_imagen(img)
                        
                        if img_array is None:
                            st.error("Error procesando la imagen")
                            return
                        
                        # Predicción
                        prediccion = modelo.predict(
                            np.expand_dims(img_array, 0), 
                            verbose=0
                        )[0][0]
                        
                        # Generar visualizaciones
                        heatmap = generar_mapa_calor(img_array, modelo)
                        overlay = crear_overlay(img_array, heatmap)
                    
                    # Mostrar resultados
                    st.markdown(f"### {IDIOMAS[idioma]['resultados']}")
                    
                    # Métrica principal
                    prob_percent = prediccion * 100
                    delta_color = "inverse" if prediccion > 0.5 else "normal"
                    
                    st.metric(
                        IDIOMAS[idioma]["prob_covid"],
                        f"{prob_percent:.1f}%",
                        delta=f"Confianza: {max(prob_percent, 100-prob_percent):.1f}%"
                    )
                    
                    # Diagnóstico
                    if prediccion > 0.5:
                        st.markdown(f"""
                        <div class="metric-container positive-result">
                            <h4>{IDIOMAS[idioma]['positivo']}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-container negative-result">
                            <h4>{IDIOMAS[idioma]['negativo']}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Visualizaciones detalladas
                st.markdown(f"## {IDIOMAS[idioma]['hallazgos']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### {IDIOMAS[idioma]['mapa_calor']}")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(heatmap)
                    ax.set_title("Regiones de Interés (Grad-CAM)")
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("### Overlay Combinado")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(overlay)
                    ax.set_title("Imagen + Mapa de Calor")
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                
                # Interpretación
                st.markdown(f"## {IDIOMAS[idioma]['interpretacion']}")
                interpretacion, descripcion = interpretar_resultado(prediccion, idioma)
                
                st.markdown(f"**{interpretacion}**")
                st.write(descripcion)
                
                # Generar y descargar reporte
                st.markdown("## 📄 Reporte")
                
                try:
                    pdf_path = generar_reporte_pdf(img, prediccion, heatmap, overlay, idioma)
                    
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    st.download_button(
                        label=IDIOMAS[idioma]["descargar_reporte"],
                        data=pdf_bytes,
                        file_name=f"reporte_covid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    # Limpiar archivo temporal
                    os.unlink(pdf_path)
                    
                except Exception as e:
                    st.error(f"Error generando reporte: {str(e)}")
        
        except Exception as e:
            st.error(f"{IDIOMAS[idioma]['error_carga']}: {str(e)}")

if __name__ == "__main__":
    main()