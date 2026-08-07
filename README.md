# 🧠 Detector de Emociones | Diario Emocional con IA

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.61-FF4B4B?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![GitHub last commit](https://img.shields.io/github/last-commit/2024-L/detector-emociones)

> Un espejo emocional personal impulsado por Inteligencia Artificial. Escribe cómo te sientes y deja que un modelo de lenguaje analice tu estado de ánimo, guarde tu historial y te acompañe en el proceso.

## ✨ Características Principales

- **Clasificación de Emociones en Español**: Detecta Alegría 😃, Tristeza 😢, Esperanza 🌟 y Neutro 😐 usando un modelo Transformer (XLM-RoBERTa).
- **Interfaz Moderna y Empática**: Diseño UI con tema personalizado (colores lavanda/violeta) y mensajes de acompañamiento según la emoción detectada.
- **Diario Interactivo**: Historial persistente en CSV con opción de descarga y borrado seguro.
- **Análisis Visual (Pestañas)**:
  - Gráficas de barras de probabilidad por emoción.
  - Evolución temporal de las emociones (gráfico de dispersión interactivo).
  - Métricas acumuladas (emoción dominante, confianza media).
- **Modelo v2 Reforzado**: Incluye un script de fine-tuning que mejora drásticamente la detección de la emoción "Esperanza".

## 🏗️ Tecnologías Utilizadas

- **Frontend**: Streamlit (con diseño responsive y tema custom).
- **Backend / IA**: PyTorch, Hugging Face `transformers`.
- **Datos**: Pandas, Altair (para visualizaciones interactivas).
- **Control de Versiones**: Git & GitHub.

## 🚀 Instalación y Uso Local

Sigue estos pasos para correr la aplicación en tu propia máquina:

### 1. Clonar el repositorio
```bash
git clone https://github.com/2024-L/detector-emociones.git
cd detector-emociones