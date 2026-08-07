# scripts/entrenar_v2.py
# Re-entrena el modelo original para reforzar "esperanza"
# y guarda el resultado en models/v2 sin tocar el proyecto original.

from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Rutas
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELO_ORIGINAL = Path(r"D:/Proyectos Nuevos/Modelación/modelo_emociones_balanceado")
SALIDA = BASE_DIR / "models" / "v2"
SALIDA.mkdir(parents=True, exist_ok=True)

EMOCIONES = ["neutro", "alegría", "esperanza", "tristeza"]

# -----------------------------
# Banco de frases de entrenamiento
# -----------------------------
DATOS = [
    # --- ESPERANZA (reforzada) ---
    ("Tengo esperanza de que todo mejorará", "esperanza"),
    ("Confío en que mañana será un buen día", "esperanza"),
    ("Espero con ilusión que llegue el fin de semana", "esperanza"),
    ("Sé que las cosas van a salir bien", "esperanza"),
    ("Tengo fe en que mi proyecto funcionará", "esperanza"),
    ("Estoy ilusionado con el futuro", "esperanza"),
    ("Pronto veré a mi familia y me hace feliz", "esperanza"),
    ("Creo que lo peor ya pasó", "esperanza"),
    ("Mañana tendré nuevas oportunidades", "esperanza"),
    ("Espero que mi entrevista de trabajo salga bien", "esperanza"),
    ("Confío en que encontraré una solución", "esperanza"),
    ("Tengo la ilusión de aprender algo nuevo cada día", "esperanza"),
    ("Sé que con esfuerzo lograré mis metas", "esperanza"),
    ("Espero que este año sea mejor que el anterior", "esperanza"),
    ("Me emociona pensar en lo que viene", "esperanza"),
    ("Tengo esperanza en que la situación mejore", "esperanza"),
    ("Confío en que mi salud se recuperará pronto", "esperanza"),
    ("Espero aprobar el examen con mi esfuerzo", "esperanza"),
    ("Creo que pronto recibiré buenas noticias", "esperanza"),
    ("Estoy esperando con ganas las vacaciones", "esperanza"),
    ("Tengo la esperanza de volver a verte pronto", "esperanza"),
    ("Sé que después de la tormenta sale el sol", "esperanza"),
    ("Espero que mi equipo gane el partido", "esperanza"),
    ("Confío en que todo se resolverá a mi favor", "esperanza"),
    ("Me ilusiona empezar este nuevo proyecto", "esperanza"),
    ("Espero que mi amigo se recupere pronto", "esperanza"),
    ("Tengo fe en que el tratamiento funcionará", "esperanza"),
    ("Creo que las cosas mejorarán con el tiempo", "esperanza"),
    ("Espero con ansias el nacimiento de mi sobrino", "esperanza"),
    ("Confío en que pronto encontraré trabajo", "esperanza"),
    ("Tengo esperanza de que nos volvamos a encontrar", "esperanza"),
    ("Sé que mi esfuerzo dará frutos", "esperanza"),
    ("Espero que mañana tengamos un día soleado", "esperanza"),
    ("Me hace ilusión planear mi viaje", "esperanza"),
    ("Tengo la esperanza de que todo se arregle", "esperanza"),
    ("Confío en que mis sueños se cumplirán", "esperanza"),
    ("Espero que este proyecto abra nuevas puertas", "esperanza"),
    ("Creo que el futuro será mejor", "esperanza"),
    ("Espero con esperanza la respuesta de la universidad", "esperanza"),
    ("Tengo fe en que mi familia saldrá adelante", "esperanza"),
    # --- ALEGRÍA ---
    ("¡Qué feliz estoy hoy!", "alegría"),
    ("Me siento muy contento por la noticia", "alegría"),
    ("¡Gané el premio, no lo puedo creer!", "alegría"),
    ("Estoy feliz de verte de nuevo", "alegría"),
    ("Hoy fue un día excelente", "alegría"),
    ("Me encanta este regalo", "alegría"),
    ("¡Qué alegría tan grande!", "alegría"),
    ("Estoy muy orgulloso de mi logro", "alegría"),
    ("Me siento eufórico por el triunfo", "alegría"),
    ("¡Felicidades por tu cumpleaños!", "alegría"),
    ("Disfruté mucho la fiesta de ayer", "alegría"),
    ("Me hace muy feliz tu visita", "alegría"),
    ("Estoy radiante de felicidad", "alegría"),
    ("¡Qué bueno que todo salió perfecto!", "alegría"),
    ("Me siento pleno y agradecido", "alegría"),
    ("Hoy celebramos con toda la familia", "alegría"),
    ("¡Estoy emocionado por la buena noticia!", "alegría"),
    ("Me divertí muchísimo en el paseo", "alegría"),
    ("Qué satisfacción haber terminado el proyecto", "alegría"),
    ("Estoy contento con mi nuevo trabajo", "alegría"),
    ("¡Bravo! Lo logramos juntos", "alegría"),
    ("Me alegra mucho tu éxito", "alegría"),
    ("Hoy amanecí con una sonrisa", "alegría"),
    ("¡Qué día tan maravilloso!", "alegría"),
    ("Me siento dichoso de tenerte en mi vida", "alegría"),
    ("Estoy feliz por haber logrado mucho hoy", "alegría"),
    ("¡Excelente! Superamos la meta", "alegría"),
    ("Me encanta compartir contigo", "alegría"),
    ("¡Qué gusto saludarte después de tanto tiempo!", "alegría"),
    ("Estoy muy animado con los resultados", "alegría"),
    ("Hoy reí como hacía tiempo no reía", "alegría"),
    ("¡Increíble! Recibí un ascenso", "alegría"),
    ("Me siento contento con lo logrado", "alegría"),
    ("¡Qué bonito detalle tuviste conmigo!", "alegría"),
    ("Estoy feliz porque mi equipo ganó", "alegría"),
    # --- TRISTEZA ---
    ("Me siento muy triste hoy", "tristeza"),
    ("Estoy deprimido por la noticia", "tristeza"),
    ("Lloré toda la noche por su partida", "tristeza"),
    ("Me siento solo y sin ánimos", "tristeza"),
    ("Nada me alegra en estos días", "tristeza"),
    ("Estoy decepcionado de todo", "tristeza"),
    ("Me duele el corazón por lo que pasó", "tristeza"),
    ("Siento un vacío enorme", "tristeza"),
    ("Estoy agotado y sin ganas de nada", "tristeza"),
    ("Me entristece recordar aquellos tiempos", "tristeza"),
    ("Hoy me siento melancólico", "tristeza"),
    ("Perdí algo muy importante para mí", "tristeza"),
    ("Me siento abatido por el fracaso", "tristeza"),
    ("Estoy triste porque mi amigo se fue", "tristeza"),
    ("No tengo ganas de salir de casa", "tristeza"),
    ("Me siento desanimado y cansado", "tristeza"),
    ("Extraño mucho a mi abuela", "tristeza"),
    ("Hoy lloré sin poder evitarlo", "tristeza"),
    ("Me siento infeliz en este momento", "tristeza"),
    ("Estoy apenado por lo ocurrido", "tristeza"),
    ("La soledad me pesa demasiado", "tristeza"),
    ("Me siento triste y sin fuerzas", "tristeza"),
    ("Todo me sale mal últimamente", "tristeza"),
    ("Estoy afligido por la pérdida", "tristeza"),
    ("Me embarga la nostalgia", "tristeza"),
    ("Siento que nadie me comprende", "tristeza"),
    ("Estoy desilusionado con la situación", "tristeza"),
    ("Me falta energía para continuar", "tristeza"),
    ("Hoy es un día gris para mí", "tristeza"),
    ("Me siento abandonado", "tristeza"),
    ("Lloro por dentro aunque no se note", "tristeza"),
    ("Estoy triste por el final de esta etapa", "tristeza"),
    ("Me apena no haber podido despedirme", "tristeza"),
    ("Siento una pena profunda", "tristeza"),
    ("Me cuesta sonreír estos días", "tristeza"),
    # --- NEUTRO ---
    ("Hoy es lunes", "neutro"),
    ("El cielo es azul", "neutro"),
    ("Voy a comprar pan al supermercado", "neutro"),
    ("La reunión es a las tres de la tarde", "neutro"),
    ("El tren sale en veinte minutos", "neutro"),
    ("Estoy escribiendo un correo", "neutro"),
    ("La temperatura hoy es de veinte grados", "neutro"),
    ("El libro tiene trescientas páginas", "neutro"),
    ("Mañana tengo que ir al banco", "neutro"),
    ("El café está sobre la mesa", "neutro"),
    ("Necesito renovar mi documento", "neutro"),
    ("La oficina cierra a las seis", "neutro"),
    ("Hoy cociné arroz con pollo", "neutro"),
    ("El semáforo estaba en rojo", "neutro"),
    ("Tengo una cita médica el jueves", "neutro"),
    ("El autobús pasa cada quince minutos", "neutro"),
    ("Dejé las llaves en la mesa", "neutro"),
    ("El informe debe entregarse el viernes", "neutro"),
    ("Estoy leyendo un artículo de tecnología", "neutro"),
    ("La película dura dos horas", "neutro"),
    ("Compré verduras en el mercado", "neutro"),
    ("El vecino tiene un carro rojo", "neutro"),
    ("Hoy hay tráfico en la avenida", "neutro"),
    ("Mi teléfono necesita carga", "neutro"),
    ("La clase empieza a las ocho", "neutro"),
    ("El paquete llegó esta mañana", "neutro"),
    ("Voy a organizar mi escritorio", "neutro"),
    ("El agua hierve a cien grados", "neutro"),
    ("Tengo que imprimir unos documentos", "neutro"),
    ("El museo abre los sábados", "neutro"),
]

# -----------------------------
# Entrenamiento
# -----------------------------
def main():
    torch.manual_seed(42)

    print("Cargando tokenizer y modelo original...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_ORIGINAL)
    model = AutoModelForSequenceClassification.from_pretrained(MODELO_ORIGINAL)

    df = pd.DataFrame(DATOS, columns=["frase", "emocion"])
    df.to_csv(BASE_DIR / "data" / "entrenamiento_v2.csv", index=False)
    print(f"Banco de frases guardado ({len(df)} frases).")

    textos = df["frase"].tolist()
    etiquetas = torch.tensor([EMOCIONES.index(e) for e in df["emocion"]])

    enc = tokenizer(textos, padding=True, truncation=True, max_length=64, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    EPOCHS = 3
    BATCH = 8
    print("Entrenando modelo v2 (puede tardar unos minutos)...")

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(textos))
        perdida_total = 0.0
        pasos = 0
        for i in range(0, len(textos), BATCH):
            idx = perm[i:i + BATCH]
            optimizer.zero_grad()
            out = model(input_ids=input_ids[idx], attention_mask=attention_mask[idx], labels=etiquetas[idx])
            out.loss.backward()
            optimizer.step()
            perdida_total += out.loss.item()
            pasos += 1
        print(f"Época {epoch + 1}/{EPOCHS} - pérdida media: {perdida_total / pasos:.3f}")

    # Guardar modelo v2 con nombres reales de emociones
    model.config.id2label = {i: e for i, e in enumerate(EMOCIONES)}
    model.config.label2id = {e: i for i, e in enumerate(EMOCIONES)}
    model.save_pretrained(SALIDA)
    tokenizer.save_pretrained(SALIDA)
    print(f"Modelo v2 guardado en: {SALIDA}")

    # Pruebas rápidas
    print("\nPruebas rápidas:")
    model.eval()
    pruebas = [
        "Tengo esperanza de que todo mejorará mañana",
        "Me siento muy triste y solo",
        "¡Qué feliz estoy hoy!",
        "Hoy es lunes y el cielo es azul",
    ]
    for p in pruebas:
        inp = tokenizer(p, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            pred = EMOCIONES[int(model(**inp).logits.argmax(-1))]
        print(f"  {p} -> {pred}")

    print("\n¡Listo! Ahora el siguiente paso es conectar la app al modelo v2.")

if __name__ == "__main__":
    main()