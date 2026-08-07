import streamlit as st
import pandas as pd
import torch
import altair as alt
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_PATH, HISTORIAL_CSV

# -----------------------------
# Configuración de la página
# -----------------------------
st.set_page_config(page_title="🧠 Detector de Emociones", page_icon="😊", layout="wide")

# -----------------------------
# Cargar el modelo
# -----------------------------
@st.cache_resource
def cargar_modelo():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = cargar_modelo()

# -----------------------------
# Datos y utilidades
# -----------------------------
emociones = ["neutro", "alegría", "esperanza", "tristeza"]

emojis = {
    "alegría": "😃",
    "neutro": "😐",
    "tristeza": "😢",
    "esperanza": "🌟",
}

colores_hex = {
    "neutro": "#6B7280",
    "alegría": "#22C55E",
    "esperanza": "#3B82F6",
    "tristeza": "#EF4444",
}

mensajes_empaticos = {
    "alegría": "¡Qué bueno leerte así! Disfruta y guarda este momento en tu memoria. 🎉",
    "tristeza": "Lamento que te sientas así. Escribirlo ya es un paso importante; sé amable contigo hoy. 💙",
    "esperanza": "Qué lindo mirar hacia adelante con ilusión. Sigue alimentando esa chispa. 🌱",
    "neutro": "Un día tranquilo también cuenta. A veces la calma es justo lo que se necesita. 😌",
}

def cargar_historial():
    if not HISTORIAL_CSV.exists():
        return pd.DataFrame(columns=["fecha", "frase", "emocion", "confianza"])
    try:
        return pd.read_csv(HISTORIAL_CSV)
    except Exception:
        return pd.DataFrame(columns=["fecha", "frase", "emocion", "confianza"])

historial = cargar_historial()

# -----------------------------
# Pestañas (Tabs)
# -----------------------------
tab1, tab2 = st.tabs(["✍️ Analizar Nueva Frase", "📊 Análisis de mi Diario"])

# ===========================
# TAB 1: ANALIZAR
# ===========================
with tab1:
    st.title("🧠 ¿Cómo te sientes hoy?")
    st.write("Escribe una frase o elige un ejemplo para comenzar.")

    if "frase" not in st.session_state:
        st.session_state["frase"] = ""

    ejemplos = [
        "😃 ¡Qué feliz estoy hoy!",
        "😢 Me siento muy triste",
        "🌟 Tengo esperanza en el mañana",
        "😐 Hoy es un día cualquiera",
    ]

    cols = st.columns(len(ejemplos))
    for col, ejemplo in zip(cols, ejemplos):
        if col.button(ejemplo):
            st.session_state["frase"] = ejemplo

    frase = st.text_input("Escribe tu frase aquí:", key="frase")

    if st.button("Analizar emoción", type="primary") and frase.strip():
        inputs = tokenizer(frase, padding=True, truncation=True, max_length=128, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = probs.argmax(-1).item()
            confianza = probs[0][pred].item()

        emocion = emociones[pred]
        emoji = emojis.get(emocion, "🙂")

        mensaje = f"{emoji} Emoción detectada: **{emocion.capitalize()}**"
        if emocion == "alegría":
            st.success(mensaje)
        elif emocion == "tristeza":
            st.error(mensaje)
        elif emocion == "esperanza":
            st.info(mensaje)
        else:
            st.warning(mensaje)

        st.progress(confianza)
        st.caption(f"Confianza: {confianza:.1%}")
        st.markdown(f"*{mensajes_empaticos.get(emocion, '')}*")

        st.subheader("📊 Probabilidades por emoción")
        df_probs = pd.DataFrame({
            "emocion": emociones,
            "probabilidad": [round(p, 3) for p in probs[0].tolist()],
        })
        
        chart_probs = alt.Chart(df_probs).mark_bar().encode(
            x=alt.X("emocion:N", title="Emoción", sort=None),
            y=alt.Y("probabilidad:Q", title="Probabilidad"),
            color=alt.Color("emocion:N", scale=alt.Scale(domain=emociones, range=[colores_hex[e] for e in emociones]), legend=None),
        )
        st.altair_chart(chart_probs, use_container_width=True)

        nueva_entrada = pd.DataFrame({
            "fecha": [datetime.now().strftime("%Y-%m-%d %H:%M")],
            "frase": [frase.strip()],
            "emocion": [emocion],
            "confianza": [round(confianza, 2)],
        })
        historial = pd.concat([historial, nueva_entrada], ignore_index=True)
        historial.to_csv(HISTORIAL_CSV, index=False)
        st.success("📁 Frase guardada en tu diario emocional.")

    st.divider()
    st.subheader("📖 Últimas 5 entradas")
    if historial.empty:
        st.info("Aún no hay entradas.")
    else:
        st.dataframe(historial.tail(5).iloc[::-1])

# ===========================
# TAB 2: ANÁLISIS DEL DIARIO
# ===========================
with tab2:
    st.title("📊 Análisis de mi Diario")
    
    if historial.empty:
        st.info("Aún no hay suficientes datos. ¡Analiza algunas frases en la primera pestaña para ver tus estadísticas aquí!")
    else:
        # Métricas principales
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de reflexiones", len(historial))
        col2.metric("Emoción dominante", historial['emocion'].mode()[0].capitalize())
        col3.metric("Confianza promedio", f"{historial['confianza'].mean():.1%}")

        st.divider()

        # Distribución general
        st.subheader("🌈 Tu paisaje emocional")
        conteo = historial["emocion"].value_counts().reindex(emociones, fill_value=0).reset_index()
        conteo.columns = ["emocion", "cantidad"]
        
        chart_conteo = alt.Chart(conteo).mark_bar().encode(
            x=alt.X("emocion:N", title="Emoción", sort=None),
            y=alt.Y("cantidad:Q", title="Número de entradas"),
            color=alt.Color("emocion:N", scale=alt.Scale(domain=emociones, range=[colores_hex[e] for e in emociones]), legend=None),
        )
        st.altair_chart(chart_conteo, use_container_width=True)

        st.divider()

        # Evolución en el tiempo
        st.subheader("📈 Evolución de tus emociones")
        st.caption("Cada punto es una frase que escribiste. Pasa el mouse por encima para leerla.")
        
        df_time = historial.copy()
        df_time['fecha_dt'] = pd.to_datetime(df_time['fecha'], errors='coerce')
        df_time = df_time.dropna(subset=['fecha_dt']).sort_values('fecha_dt')

        if not df_time.empty:
            chart_time = alt.Chart(df_time).mark_circle(size=100).encode(
                x=alt.X('fecha_dt:T', title='Fecha y Hora'),
                y=alt.Y('confianza:Q', title='Confianza del modelo', scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('emocion:N', scale=alt.Scale(domain=emociones, range=[colores_hex[e] for e in emociones]), legend=alt.Legend(title="Emoción")),
                tooltip=['fecha', 'frase', 'emocion', 'confianza']
            ).properties(height=400)
            st.altair_chart(chart_time, use_container_width=True)
        else:
            st.warning("No se pudieron leer las fechas para el gráfico de evolución.")

        st.divider()

        # Descarga y Borrado
        col_descarga, col_borrar = st.columns(2)
        with col_descarga:
            csv = historial.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar diario completo (CSV)",
                data=csv,
                file_name="diario_emocional.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_borrar:
            if "confirmar_borrado" not in st.session_state:
                st.session_state["confirmar_borrado"] = False

            if st.button("🗑️ Borrar todo el historial", use_container_width=True):
                st.session_state["confirmar_borrado"] = True

            if st.session_state["confirmar_borrado"]:
                st.warning("¿Seguro que quieres borrar todo el diario? Esta acción no se puede deshacer.")
                ca, cb = st.columns(2)
                if ca.button("Sí, borrar todo"):
                    HISTORIAL_CSV.unlink(missing_ok=True)
                    st.session_state["confirmar_borrado"] = False
                    st.rerun()
                if cb.button("No, conservar"):
                    st.session_state["confirmar_borrado"] = False
                    st.rerun()

# -----------------------------
# Barra lateral
# -----------------------------
with st.sidebar:
    st.title("🧠 Mi Diario IA")
    st.caption("Modelo v2 (Refuerzo en Esperanza)")
    st.divider()
    st.write("**Emociones que detecto:**")
    st.write("😃 Alegría")
    st.write("😢 Tristeza")
    st.write("🌟 Esperanza")
    st.write("😐 Neutro")