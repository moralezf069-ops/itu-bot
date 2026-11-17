import os
import logging
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from google import genai
from google.genai import types

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configurar logging para ver la terminal claramente (necesario para el Hito 3)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Obtener tokens de seguridad
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Inicializar cliente oficial de Google GenAI
client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# FUNCIÓN PROPIA (TOOL USE / FUNCTION CALLING)
# ==========================================
def consultar_clima(ciudad: str = "Mendoza") -> str:
    """Consulta el clima actual en una ciudad específica (por defecto Mendoza) para los estudiantes del ITU.
    
    Args:
        ciudad: El nombre de la ciudad a consultar.
    """
    try:
        logger.info(f"-> [TOOL CALL] Ejecutando consultar_clima para la ciudad: {ciudad}")
        url = f"https://wttr.in/{ciudad}?format=3"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.text.strip()
        else:
            return "No se pudo obtener el clima en este momento debido a un error en el servidor externo."
    except Exception as e:
        logger.error(f"Error en consultar_clima: {str(e)}")
        return f"Ocurrió un error al conectar con el servicio de clima: {str(e)}"

# ==========================================
# PROMPT DE SISTEMA (PERSONALIDAD DE MALBI)
# ==========================================
SYSTEM_PROMPT = """
Eres Malbi, la asistente virtual oficial del Instituto Tecnológico Universitario (ITU).
Tu rol es ayudar y orientar exclusivamente a los estudiantes de la Tecnicatura Universitaria en Desarrollo de Software.
Personalidad y directrices:
- Identidad: Saluda amablemente, reconoce a los estudiantes como colegas de desarrollo de software y mantén un tono cercano, profesional y entusiasta.
- Alcance: Estás enfocada en temas académicos, programación (Python, JavaScript, etc.), bases de datos, Git/GitHub y orientación del ITU. Si te consultan cosas totalmente ajenas (como recetas de cocina o temas personales), pon límites con amabilidad aclarando que estás para ayudar con el cursado, o da una respuesta breve sin perder el foco académico.
- Honestidad: Si te consultan datos cambiantes o administrativos exactos (como montos de cuotas o fechas específicas de exámenes que no posees), evita alucinaciones y deriva siempre a bedelía o canales oficiales del ITU.
- Estilo: Usa un formato claro, usa viñetas cuando sea útil y mantén respuestas dinámicas.
"""

# Memoria conversacional simple en memoria / diccionario por chat_id
historiales = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start para iniciar la interacción."""
    user_name = update.effective_user.first_name
    await update.message.reply_text(
        f"¡Hola {user_name}! Soy Malbi, tu asistente virtual del ITU. ¿En qué te puedo ayudar hoy con tu cursado o desarrollo de software?"
    )

async def manejar_mensaje(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja los mensajes de texto de los usuarios utilizando Gemini y herramientas."""
    user_id = update.effective_user.id
    user_message = update.message.text

    # Inicializar historial del usuario si no existe
    if user_id not in historiales:
        historiales[user_id] = []

    # Agregar el mensaje del usuario al historial
    historiales[user_id].append(f"Usuario: {user_message}")

    # Construir el contexto acumulado de la conversación para la memoria
    contexto_conversacion = "\n".join(historiales[user_id][-10:]) # Últimos 10 mensajes

    prompt_completo = f"""
    Historial de la conversación reciente:
    {contexto_conversacion}

    Responde al último mensaje del usuario manteniendo tu personalidad institucional de Malbi.
    """

    try:
        # Llamada a Gemini usando el SDK moderno y pasando la función propia en tools
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt_completo,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[consultar_clima],  # <--- Acá se habilita la función propia
                temperature=0.7,
            )
        )

        # Verificar si el modelo decidió invocar la función propia (Tool Call)
        if response.function_calls:
            for function_call in response.function_calls:
                logger.info(f"Gemini solicitó llamar a la función: {function_call.name} con argumentos: {function_call.args}")
                
                # Ejecutar localmente la función solicitada
                if function_call.name == "consultar_clima":
                    args = function_call.args or {}
                    ciudad_arg = args.get("ciudad", "Mendoza")
                    resultado_funcion = consultar_clima(ciudad=ciudad_arg)
                    
                    # Enviar el resultado de vuelta a Gemini para que redacte la respuesta final al usuario
                    response_final = client.models.generate_content(
                        model='gemini-3.5-flash-lite',
                        contents=f"El resultado de la función consultar_clima fue: {resultado_funcion}. Responde al usuario amablemente incorporando este dato.",
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.7,
                        )
                    )
                    bot_reply = response_final.text
                else:
                    bot_reply = "Ejecuté la herramienta solicitada pero no se reconoció la función."
        else:
            bot_reply = response.text

        # Guardar respuesta en el historial
        historiales[user_id].append(f"Malbi: {bot_reply}")

        # Enviar respuesta a Telegram
        await update.message.reply_text(bot_reply)

    except Exception as e:
        logger.error(f"Error al conectar con la API de Gemini: {str(e)}")
        await update.message.reply_text("Disculpa, tuve un problema técnico procesando tu consulta. Por favor, intenta de nuevo en unos momentos.")

def main():
    """Función principal para arrancar el bot."""
    if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
        logger.error("Faltan configurar TELEGRAM_TOKEN o GEMINI_API_KEY en el archivo .env")
        return

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), manejar_mensaje))

    logger.info("¡Bot Malbi iniciado correctamente y listo para el Hito 3!")
    application.run_polling()

if __name__ == "__main__":
    main()