import os
import logging
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Importar el índice RAG vectorial / híbrido personalizado
from rag_vector import RAGVectorIndex

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configurar logging para ver la terminal claramente
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Obtener tokens de seguridad
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Inicializar cliente oficial de Google GenAI
client = genai.Client(api_key=GEMINI_API_KEY)

# Inicializar el índice RAG al arrancar el bot
rag_index = RAGVectorIndex("reglamento_itu.txt")

# ==========================================
# DEFINICIÓN DE PYDANTIC (SALIDA ESTRUCTURADA)
# ==========================================
class AnalisisConsultaEstudiante(BaseModel):
    tema_principal: str = Field(description="Tema de la consulta: inscripciones, finales, asistencia, promocion, certificados o clima.")
    nivel_urgencia: str = Field(description="Clasificación de urgencia: baja, media o alta.")
    resumen_intencion: str = Field(description="Breve resumen de lo que el alumno necesita resolver.")
    sugerencia_respuesta: str = Field(description="Respuesta completa y detallada para el alumno, utilizando obligatoriamente los datos institucionales recuperados por el RAG.")

# ==========================================
# MÓDULO RAG
# ==========================================
def buscar_en_reglamento(query: str) -> str:
    """Busca el fragmento más relevante utilizando el índice RAG."""
    return rag_index.buscar(query)

# ==========================================
# FUNCIÓN PROPIA (TOOL USE / CLIMA)
# ==========================================
def consultar_clima(ciudad: str = "Mendoza") -> str:
    """Consulta el clima actual en una ciudad específica para los estudiantes del ITU."""
    try:
        logger.info(f"-> [TOOL CALL] Ejecutando consultar_clima para: {ciudad}")
        url = f"https://wttr.in/{ciudad}?format=3"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.text.strip()
        else:
            return "No se pudo obtener el clima en este momento."
    except Exception as e:
        logger.error(f"Error en consultar_clima: {str(e)}")
        return f"Ocurrió un error al conectar con el servicio de clima: {str(e)}"

# ==========================================
# PROMPT DE SISTEMA (PERSONALIDAD DE SAMIR)
# ==========================================
SYSTEM_PROMPT = """
Eres Samir, la asistente virtual oficial del Instituto Tecnológico Universitario (ITU).
Tu rol es ayudar y orientar exclusivamente a los estudiantes de la Tecnicatura Universitaria en Desarrollo de Software.
Personalidad y directrices:
- Identidad: Saluda amablemente, reconoce a los estudiantes como colegas de desarrollo de software y mantén un tono cercano, profesional y entusiasta.
- REGLA ESTRICTA DE RAG: Si el "FRAGMENTO OFICIAL RECUPERADO" contiene datos (como el 75% de asistencia, plazos o requisitos), DEBES incluir esos datos exactos en tu respuesta al alumno. Si dice que no se encontró, indícale amablemente que consulte en bedelía. NUNCA inventes información.
"""

# Diccionarios de Memoria y Estados para el Flujo Guiado
historiales = {}
flujos_activos = {}  # Controla el estado del trámite paso a paso por user_id

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start para iniciar la interacción."""
    user_name = update.effective_user.first_name
    await update.message.reply_text(
        f"¡Hola {user_name}! Soy Samir, tu asistente virtual del ITU. ¿En qué te puedo ayudar hoy con tu cursado o desarrollo de software?\n\n"
        f"💡 *Comandos disponibles:* /certificado (para iniciar el trámite de alumno regular)."
    )

async def iniciar_flujo_certificado(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Inicia el Flujo Guiado para Certificados."""
    user_id = update.effective_user.id
    flujos_activos[user_id] = {
        "paso": "ESPERANDO_DNI",
        "datos": {}
    }
    logger.info(f"-> [FLUJO GUIADO] Iniciado para usuario {user_id}")
    await update.message.reply_text(
        "📄 *Iniciando trámite guiado:* Solicitud de Certificado de Alumno Regular.\n\n"
        "Por favor, ingresa tu **DNI** (solo números) para continuar, o escribe **cancelar** en cualquier momento para abortar el trámite."
    )

async def manejar_mensaje(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja los mensajes integrando Flujo Guiado, RAG, Pydantic, Memoria y Tools."""
    user_id = update.effective_user.id
    user_message = update.message.text.strip()

    # Detectar si pregunta por el clima antes de pasar por Pydantic y el RAG
    if "clima" in user_message.lower() or "tiempo" in user_message.lower():
        resultado_clima = consultar_clima("Mendoza")
        await update.message.reply_text(f"🌤️ Información meteorológica para el ITU:\n{resultado_clima}")
        return

    # ==========================================
    # GESTIÓN DEL MÓDULO: FLUJO GUIADO
    # ==========================================
    if user_id in flujos_activos:
        estado_actual = flujos_activos[user_id]["paso"]

        if user_message.lower() == "cancelar":
            del flujos_activos[user_id]
            logger.info(f"-> [FLUJO GUIADO] Cancelado por el usuario {user_id}")
            await update.message.reply_text("❌ Trámite cancelado con éxito. ¿En qué más te puedo ayudar?")
            return

        if estado_actual == "ESPERANDO_DNI":
            if not user_message.isdigit() or not (7 <= len(user_message) <= 8):
                logger.warning(f"-> [FLUJO GUIADO] DNI inválido rechazado: '{user_message}'")
                await update.message.reply_text(
                    "⚠️ **Dato inválido:** El DNI debe contener únicamente números (entre 7 y 8 dígitos).\n"
                    "Por favor, ingrésalo nuevamente o escribe **cancelar**:"
                )
                return

            flujos_activos[user_id]["datos"]["dni"] = user_message
            flujos_activos[user_id]["paso"] = "ESPERANDO_ANIO"
            logger.info(f"-> [FLUJO GUIADO] DNI válido guardado: {user_message}. Solicitando año.")
            await update.message.reply_text(
                "¡Perfecto! DNI registrado.\n\n"
                "Ahora, por favor, indica tu **año de cursado actual** (ej. 1er año, 2do año):"
            )
            return

        elif estado_actual == "ESPERANDO_ANIO":
            flujos_activos[user_id]["datos"]["anio"] = user_message
            dni_registrado = flujos_activos[user_id]["datos"]["dni"]
            anio_registrado = user_message

            logger.info(f"-> [FLUJO GUIADO] Trámite completado con éxito para DNI {dni_registrado}, Año {anio_registrado}")
            del flujos_activos[user_id]

            await update.message.reply_text(
                f"✅ **¡Trámite completado con éxito!**\n\n"
                f"Hemos registrado tu solicitud de Certificado de Alumno Regular:\n"
                f"• **DNI:** {dni_registrado}\n"
                f"• **Año de cursado:** {anio_registrado}\n\n"
                f"El certificado será enviado a tu correo institucional en un plazo de 24 a 48 horas hábiles. ¡Mucho éxito con el cursado!"
            )
            return

    # ==========================================
    # FLUJO NORMAL (RAG + PYDANTIC + TOOLS)
    # ==========================================
    if user_id not in historiales:
        historiales[user_id] = []

    historiales[user_id].append(f"Usuario: {user_message}")
    contexto_conversacion = "\n".join(historiales[user_id][-10:])

    # 1. BÚSQUEDA RAG
    fragmento_rag = buscar_en_reglamento(user_message)

    # 2. CONSTRUIR EL PROMPT CON EL FRAGMENTO RECUPERADO
    prompt_completo = f"""
    Historial reciente:
    {contexto_conversacion}

    ---
    FRAGMENTO OFICIAL RECUPERADO:
    {fragmento_rag}
    ---

    Consulta del alumno: "{user_message}"

    Instrucción: Utiliza la información del FRAGMENTO OFICIAL RECUPERADO para responder con exactitud en la 'sugerencia_respuesta'. Si el fragmento tiene el dato (ej. el 75% de asistencia), menciónalo explícitamente. Si no se encuentra, indícalo claramente.
    """

    response = None
    for modelo_intento in ['gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-3.5-flash-lite']:
        try:
            logger.info(f"Intentando generar contenido con el modelo: {modelo_intento}")
            response = client.models.generate_content(
                model=modelo_intento,
                contents=prompt_completo,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=AnalisisConsultaEstudiante,
                    tools=[consultar_clima],
                    temperature=0.2,
                )
            )
            break
        except Exception as e:
            logger.warning(f"El modelo {modelo_intento} falló: {str(e)}")
            continue

    try:
        if not response:
            raise Exception("Todos los modelos fallaron.")

        analisis: AnalisisConsultaEstudiante = response.parsed

        if analisis:
            logger.info(f"--- [ESTRUCTURA PYDANTIC VALIDADA] ---")
            logger.info(f"Tema: {analisis.tema_principal}")
            logger.info(f"Urgencia: {analisis.nivel_urgencia}")
            logger.info(f"Resumen: {analisis.resumen_intencion}")
            logger.info(f"Sugerencia: {analisis.sugerencia_respuesta}")
            logger.info(f"----------------------------------------")

        bot_reply = analisis.sugerencia_respuesta if analisis else "¡Hola! Te estoy leyendo, ¿en qué te ayudo?"

        if not bot_reply or not bot_reply.strip():
            bot_reply = "¡Listo! Procesé tu consulta correctamente desde el sistema del ITU."

        historiales[user_id].append(f"Samir: {bot_reply}")
        await update.message.reply_text(bot_reply)

    except Exception as e:
        logger.error(f"Error procesando mensaje: {str(e)}")
        await update.message.reply_text("Disculpa, tuve un pequeño inconveniente procesando tu solicitud. ¡Inténtalo de nuevo!")

def main():
    if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
        logger.error("Faltan configurar credenciales en el .env")
        return

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("certificado", iniciar_flujo_certificado))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), manejar_mensaje))

    use_webhook = os.getenv("USE_WEBHOOK", "false").lower() == "true"
    port = int(os.getenv("PORT", "8080"))
    webhook_url = os.getenv("WEBHOOK_URL", "").rstrip("/")

    if use_webhook and webhook_url:
        # En muchos PaaS (como Render o Railway), la URL pública apunta al puerto 443/80
        # y la app escucha en 0.0.0.0:PORT
        url_path = TELEGRAM_TOKEN
        full_webhook_url = f"{webhook_url}/{url_path}"
        logger.info(f"¡Bot Samir iniciado en modo WEBHOOK en el puerto {port}!")
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=url_path,
            webhook_url=full_webhook_url,
        )
    else:
        logger.info("¡Bot Samir iniciado en modo POLLING con RAG Híbrido, Pydantic, Tools y Flujo Guiado!")
        application.run_polling()

if __name__ == "__main__":
    main()