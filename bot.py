import os
import logging
import json
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

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

# ==========================================
# DEFINICIÓN DE PYDANTIC (SALIDA ESTRUCTURADA)
# ==========================================
class AnalisisConsultaEstudiante(BaseModel):
    tema_principal: str = Field(description="Tema de la consulta: inscripciones, finales, asistencia, promocion, certificados o clima.")
    nivel_urgencia: str = Field(description="Clasificación de urgencia: baja, media o alta.")
    resumen_intencion: str = Field(description="Breve resumen de lo que el alumno necesita resolver.")
    sugerencia_respuesta: str = Field(description="Respuesta completa y detallada para el alumno, utilizando obligatoriamente los datos del documento institucional (RAG).")

# ==========================================
# MÓDULO RAG (BÚSQUEDA EN DOCUMENTOS DEL ITU)
# ==========================================
def buscar_en_reglamento(query: str) -> str:
    """Carga el documento institucional completo para asegurar que Gemini tenga todo el contexto disponible."""
    ruta_archivo = "reglamento_itu.txt"
    if not os.path.exists(ruta_archivo):
        logger.warning("El archivo reglamento_itu.txt no se encuentra.")
        return "No hay documentos institucionales cargados."

    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            contenido = f.read()

        logger.info(f"-> [RAG CREADO] Documento institucional cargado correctamente para la consulta: '{query}'")
        return contenido

    except Exception as e:
        logger.error(f"Error en RAG: {str(e)}")
        return "Error al procesar los documentos institucionales."

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
# PROMPT DE SISTEMA (PERSONALIDAD DE MALBI)
# ==========================================
SYSTEM_PROMPT = """
Eres Malbi, la asistente virtual oficial del Instituto Tecnológico Universitario (ITU).
Tu rol es ayudar y orientar exclusivamente a los estudiantes de la Tecnicatura Universitaria en Desarrollo de Software.
Personalidad y directrices:
- Identidad: Saluda amablemente, reconoce a los estudiantes como colegas de desarrollo de software y mantén un tono cercano, profesional y entusiasta.
- REGLA ESTRICTA DE RAG: Si la información solicitada se encuentra en el "DOCUMENTO INSTITUCIONAL COMPLETO", DEBES extraer los datos exactos (como el 75% de asistencia, plazos o requisitos) y dárselos al alumno. Si el dato no figura en absoluto, indícale amablemente que consulte en bedelía. NUNCA inventes información.
"""

# Diccionarios de Memoria y Estados para el Flujo Guiado
historiales = {}
flujos_activos = {}  # Controla el estado del trámite paso a paso por user_id

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start para iniciar la interacción."""
    user_name = update.effective_user.first_name
    await update.message.reply_text(
        f"¡Hola {user_name}! Soy Malbi, tu asistente virtual del ITU. ¿En qué te puedo ayudar hoy con tu cursado o desarrollo de software?\n\n"
        f"💡 *Comandos disponibles:* /certificado (para iniciar el trámite de alumno regular)."
    )

async def iniciar_flujo_certificado(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Inicia el Módulo Avanzado 2: Flujo Guiado para Certificados."""
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

    # ==========================================
    # GESTIÓN DEL MÓDULO 2: FLUJO GUIADO
    # ==========================================
    if user_id in flujos_activos:
        estado_actual = flujos_activos[user_id]["paso"]

        # Permitir cancelar a mitad de flujo
        if user_message.lower() == "cancelar":
            del flujos_activos[user_id]
            logger.info(f"-> [FLUJO GUIADO] Cancelado por el usuario {user_id}")
            await update.message.reply_text("❌ Trámite cancelado con éxito. ¿En qué más te puedo ayudar?")
            return

        if estado_actual == "ESPERANDO_DNI":
            # Validación de dato inválido (debe ser numérico y tener entre 7 y 8 dígitos)
            if not user_message.isdigit() or not (7 <= len(user_message) <= 8):
                logger.warning(f"-> [FLUJO GUIADO] DNI inválido rechazado: '{user_message}'")
                await update.message.reply_text(
                    "⚠️ **Dato inválido:** El DNI debe contener únicamente números (entre 7 y 8 dígitos).\n"
                    "Por favor, ingrésalo nuevamente o escribe **cancelar**:"
                )
                return
            
            # Guardar DNI y pasar al siguiente paso
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

            # Fin del flujo guiado exitoso
            logger.info(f"-> [FLUJO GUIADO] Trámite completado con éxito para DNI {dni_registrado}, Año {anio_registrado}")
            del flujos_activos[user_id] # Limpiar estado

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

    # Cargar documento RAG
    documento_rag = buscar_en_reglamento(user_message)

    prompt_completo = f"""
    Historial reciente:
    {contexto_conversacion}

    ---
    DOCUMENTO INSTITUCIONAL COMPLETO (RAG):
    {documento_rag}
    ---

    Consulta del alumno: "{user_message}"
    
    Instrucción: Lee detenidamente el documento institucional completo y responde con absoluta precisión en la 'sugerencia_respuesta' utilizando los datos reales (por ejemplo, el 75% de asistencia si te preguntan por asistencia). Si el alumno pregunta por certificados, recuérdale que puede usar el comando /certificado.
    """

    response = None
    for modelo_intento in ['gemini-3.5-flash-lite', 'gemini-1.5-flash']:
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

        # Verificar si requiere herramienta (clima)
        if response.function_calls:
            for function_call in response.function_calls:
                logger.info(f"Gemini invocó la herramienta: {function_call.name}")
                if function_call.name == "consultar_clima":
                    args = function_call.args or {}
                    ciudad_arg = args.get("ciudad", "Mendoza")
                    resultado_funcion = consultar_clima(ciudad=ciudad_arg)
                    
                    response_final = client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=f"El resultado de la función consultar_clima fue: {resultado_funcion}. Responde amablemente al usuario.",
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.7,
                        )
                    )
                    bot_reply = response_final.text
                else:
                    bot_reply = analisis.sugerencia_respuesta if analisis else "Procesado correctamente."
        else:
            bot_reply = analisis.sugerencia_respuesta if analisis else "¡Hola! Te estoy leyendo, ¿en qué te ayudo?"

        if not bot_reply or not bot_reply.strip():
            bot_reply = "¡Listo! Procesé tu consulta correctamente desde el sistema del ITU."

        historiales[user_id].append(f"Malbi: {bot_reply}")
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

    logger.info("¡Bot Malbi Hito 4 iniciado con RAG, Pydantic, Tools y Flujo Guiado!")
    application.run_polling()

if __name__ == "__main__":
    main()