import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from datetime import datetime

# Configurar la página
st.set_page_config(page_title="🧠 Detector de Emociones", page_icon="😊")
st.title("🧠 ¿Cómo te sientes hoy?")
st.write("Escribe una frase y te diré qué emoción refleja.")

# Cargar el modelo (con caché para que no se recargue cada vez)
@st.cache_resource
def cargar_modelo():
    modelo_path = "./modelo_emociones_balanceado"
    tokenizer = AutoTokenizer.from_pretrained(modelo_path)
    model = AutoModelForSequenceClassification.from_pretrained(modelo_path)
    model.eval()
    emociones = ["neutro", "alegría", "esperanza", "tristeza"]
    return tokenizer, model, emociones

tokenizer, model, emociones = cargar_modelo()

# Entrada del usuario
frase = st.text_input("Escribe tu frase aquí:", placeholder="Ej: Hoy me siento lleno de energía")

if st.button("Analizar emoción") and frase:
    # Tokenizar y predecir
    inputs = tokenizer(frase, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(-1).item()
    emocion = emociones[pred]
   
    # Mostrar resultado con emoji y color
    colores = {"alegría": "verde", "neutro": "gris", "tristeza": "rojo", "esperanza": "azul"}
    emojis = {"alegría": "😃", "neutro": "😐", "tristeza": "😢", "esperanza": "🌟"}
    st.markdown(f"### {emojis[emocion]} Emoción detectada: **{emocion.capitalize()}**",
                unsafe_allow_html=False)
   
    # Guardar historial (opcional)
    historial = pd.DataFrame({"fecha": [datetime.now().strftime("%Y-%m-%d %H:%M")],
                              "frase": [frase],
                              "emocion": [emocion]})
    try:
        historial_prev = pd.read_csv("diario_emocional.csv")
        historial = pd.concat([historial_prev, historial], ignore_index=True)
    except FileNotFoundError:
        pass
    historial.to_csv("diario_emocional.csv", index=False)
    st.info("📁 Frase guardada en tu diario emocional.")

# Mostrar historial reciente
try:
    historial_prev = pd.read_csv("diario_emocional.csv")
    st.subheader("📖 Últimas entradas de tu diario")
    st.dataframe(historial_prev.tail(5))
except:
    pass