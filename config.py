import os
from pathlib import Path

# 1) Modelo en Hugging Face (siempre disponible en internet)
MODELO_HF = "Lino2026/detector-emociones-v2"

# 2) Modelo v2 local (entrenado en esta computadora)
MODELO_V2_LOCAL = Path(__file__).parent / "models" / "v2"

# 3) Modelo original (proyecto viejo, como respaldo)
RUTA_ORIGINAL = os.getenv(
    "RUTA_MODELO",
    r"D:/Proyectos Nuevos/Modelación/modelo_emociones_balanceado",
)

# Prioridad: nube > local > original
if MODELO_V2_LOCAL.exists() and (MODELO_V2_LOCAL / "config.json").exists():
    MODEL_PATH = MODELO_V2_LOCAL
else:
    MODEL_PATH = MODELO_HF

# Datos del proyecto
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
HISTORIAL_CSV = DATA_DIR / "diario_emocional.csv"
FEEDBACK_CSV = DATA_DIR / "feedback.csv"