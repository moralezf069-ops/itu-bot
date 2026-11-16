import os
import sqlite3
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
from prompts import SYSTEM_PROMPT

# Cargar variables de entorno
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Configurar logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Inicializar cliente de Gemini con el SDK moderno
client = genai.Client(api_key=GEMINI_API_KEY)

# --- CONFIGURACIÓN DE SQLITE (Memoria Persistente) ---
def init_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

def guardar_mensaje(user_id, role, content):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
    conn.commit()
    conn.close()

def obtener_historial(user_id):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    contents = []
    for role, content in rows:
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=content)]
            )
        )
    return contents

def limpiar_historial_db(user_id):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

# --- MANEJADORES DE COMANDOS Y MENSAJES ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    limpiar_historial_db(user_id)
    nombre = update.effective_user.first_name
    await update.message.reply_text(f"¡Hola, {nombre}! Soy Malbi, tu asistente virtual del ITU. ¿En qué te puedo ayudar hoy?")

async def limpiar_conversacion(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    limpiar_historial_db(user_id)
    await update.message.reply_text("🧹 ¡Listo! He borrado el historial de nuestra conversación.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_message = update.message.text
    
    # 1. Guardar mensaje del usuario en SQLite
    guardar_mensaje(user_id, "user", user_message)
    
    # 2. Recuperar el historial
    chat_history = obtener_historial(user_id)
    
    try:
        # 3. Llamar a la API de Gemini con el modelo oficial gemini-3.8-flash
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=chat_history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
            ),
        )
        bot_response = response.text
        
        # 4. Guardar respuesta del bot en SQLite
        guardar_mensaje(user_id, "model", bot_response)
        
        await update.message.reply_text(bot_response)
        
    except Exception as e:
        logger.error(f"Error al conectar con Gemini: {e}")
        await update.message.reply_text("Disculpa, los servidores están experimentando alta demanda en este momento. Intenta enviarme el mensaje de nuevo en unos segundos.")

def main() -> None:
    init_db()
    if not TOKEN:
        raise ValueError("Falta el TELEGRAM_TOKEN en el archivo .env")
        
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("limpiar", limpiar_conversacion))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🤖 ¡Bot v3 con Gemini y SQLite iniciado con éxito!")
    application.run_polling()

if __name__ == "__main__":
    main()