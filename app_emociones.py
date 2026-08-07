import streamlit as st
import pandas as pd
import torch
import altair as alt
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_PATH, HISTORIAL_CSV, FEEDBACK_CSV

st.set_page_config(page_title="🧠 Detector de Emociones", page_icon="😊", layout="wide")

@st.cache_resource
def cargar_modelo():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = cargar_modelo()
emociones = ["neutro", "alegría", "esperanza", "tristeza"]
emojis = {"alegría": "😃", "neutro": "😐", "tristeza": "😢", "esperanza": "🌟"}
colores_hex = {"neutro": "#6B7280", "alegría": "#22C55E", "esperanza": "#3B82F6", "tristeza": "#EF4444"}
mensajes_empaticos = {
    "alegría": "¡Qué bueno leerte así! Disfruta y guarda este momento. 🎉",
    "tristeza": "Lamento que te sientas así. Escribirlo ya es un paso importante. 💙",
    "esperanza": "Qué lindo mirar hacia adelante con ilusión. 🌱",
    "neutro": "Un día tranquilo también cuenta. 😌",
}

def cargar_historial():
    if not HISTORIAL_CSV.exists():
        return pd.DataFrame(columns=["fecha", "frase", "emocion", "confianza"])
    try: return pd.read_csv(HISTORIAL_CSV)
    except: return pd.DataFrame(columns=["fecha", "frase", "emocion", "confianza"])

historial = cargar_historial()
tab1, tab2 = st.tabs(["✍️ Analizar Nueva Frase", "📊 Análisis de mi Diario"])

with tab1:
    st.title("🧠 ¿Cómo te sientes hoy?")
    st.write("Escribe una frase o elige un ejemplo.")
    if "frase" not in st.session_state: st.session_state["frase"] = ""
    ejemplos = ["😃 ¡Qué feliz estoy!", "😢 Me siento triste", "🌟 Tengo esperanza", "😐 Día cualquiera"]
    cols = st.columns(len(ejemplos))
    for col, ejemplo in zip(cols, ejemplos):
        if col.button(ejemplo): st.session_state["frase"] = ejemplo
    
    frase = st.text_input("Escribe tu frase:", key="frase")
    
    if st.button("Analizar emoción", type="primary") and frase.strip():
        inputs = tokenizer(frase, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = probs.argmax(-1).item()
            confianza = probs[0][pred].item()
        
        emocion = emociones[pred]
        emoji = emojis.get(emocion, "🙂")
        
        if emocion == "alegría": st.success(f"{emoji} Emoción: **{emocion.capitalize()}**")
        elif emocion == "tristeza": st.error(f"{emoji} Emoción: **{emocion.capitalize()}**")
        elif emocion == "esperanza": st.info(f"{emoji} Emoción: **{emocion.capitalize()}**")
        else: st.warning(f"{emoji} Emoción: **{emocion.capitalize()}**")
        
        st.progress(confianza)
        st.caption(f"Confianza: {confianza:.1%}")
        st.markdown(f"*{mensajes_empaticos.get(emocion, '')}*")
        
        st.subheader("📊 Probabilidades")
        df_probs = pd.DataFrame({"emocion": emociones, "probabilidad": [round(p, 3) for p in probs[0].tolist()]})
        chart = alt.Chart(df_probs).mark_bar().encode(
            x=alt.X("emocion:N"), y=alt.Y("probabilidad:Q"),
            color=alt.Color("emocion:N", scale=alt.Scale(domain=emociones, range=[colores_hex[e] for e in emociones]), legend=None)
        )
        st.altair_chart(chart, use_container_width=True)
        
        # Guardar en historial
        nueva = pd.DataFrame({"fecha": [datetime.now().strftime("%Y-%m-%d %H:%M")], "frase": [frase.strip()], "emocion": [emocion], "confianza": [round(confianza, 2)]})
        historial = pd.concat([historial, nueva], ignore_index=True)
        historial.to_csv(HISTORIAL_CSV, index=False)
        st.success("📁 Guardado en tu diario.")
        
        # BOTÓN DE FEEDBACK
        st.divider()
        st.subheader("💬 ¿Acerté con tu emoción?")
        col1, col2 = st.columns(2)
        if col1.button("✅ Sí, acertaste"):
            fb = pd.DataFrame({"fecha": [datetime.now().strftime("%Y-%m-%d %H:%M")], "frase": [frase.strip()], "emocion_predicha": [emocion], "correcto": [True]})
            try: fb_prev = pd.read_csv(FEEDBACK_CSV); fb_total = pd.concat([fb_prev, fb], ignore_index=True)
            except: fb_total = fb
            fb_total.to_csv(FEEDBACK_CSV, index=False)
            st.success("¡Gracias! Tu feedback ayuda a mejorar el modelo.")
        if col2.button("❌ No, fue otra"):
            emocion_real = st.selectbox("¿Cuál era la emoción correcta?", emociones)
            if st.button("Enviar corrección"):
                fb = pd.DataFrame({"fecha": [datetime.now().strftime("%Y-%m-%d %H:%M")], "frase": [frase.strip()], "emocion_predicha": [emocion], "emocion_real": [emocion_real], "correcto": [False]})
                try: fb_prev = pd.read_csv(FEEDBACK_CSV); fb_total = pd.concat([fb_prev, fb], ignore_index=True)
                except: fb_total = fb
                fb_total.to_csv(FEEDBACK_CSV, index=False)
                st.success("¡Gracias! Con esto entrenaré el modelo v3.")

with tab2:
    st.title("📊 Análisis de mi Diario")
    if historial.empty:
        st.info("Aún no hay datos.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(historial))
        c2.metric("Dominante", historial['emocion'].mode()[0].capitalize())
        c3.metric("Confianza media", f"{historial['confianza'].mean():.1%}")
        
        conteo = historial["emocion"].value_counts().reindex(emociones, fill_value=0).reset_index()
        conteo.columns = ["emocion", "cantidad"]
        chart2 = alt.Chart(conteo).mark_bar().encode(
            x=alt.X("emocion:N"), y=alt.Y("cantidad:Q"),
            color=alt.Color("emocion:N", scale=alt.Scale(domain=emociones, range=[colores_hex[e] for e in emociones]), legend=None)
        )
        st.altair_chart(chart2, use_container_width=True)
        
        col_desc, col_bor = st.columns(2)
        with col_desc:
            csv = historial.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar diario", data=csv, file_name="diario.csv", mime="text/csv")
        with col_bor:
            if st.button("🗑️ Borrar historial"):
                HISTORIAL_CSV.unlink(missing_ok=True)
                st.rerun()

with st.sidebar:
    st.title("🧠 Mi Diario IA")
    st.caption("Modelo v2 en la nube")
    st.write(f"Fuente: `Lino2026/detector-emociones-v2`")