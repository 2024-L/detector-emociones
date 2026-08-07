from pathlib import Path

# Modelo original (solo lectura)
MODELO_ORIGINAL_PATH = Path(r"D:/Proyectos Nuevos/Modelación/modelo_emociones_balanceado")

# Modelo v2 (entrenado con refuerzo de esperanza)
MODELO_V2_PATH = Path(__file__).parent / "models" / "v2"

# Por defecto usamos el v2
MODEL_PATH = MODELO_V2_PATH if MODELO_V2_PATH.exists() else MODELO_ORIGINAL_PATH

# Carpeta data del proyecto nuevo
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Archivo de historial del proyecto nuevo
HISTORIAL_CSV = DATA_DIR / "diario_emocional.csv"