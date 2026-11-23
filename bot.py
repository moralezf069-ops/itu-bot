import asyncio
import logging
import os
import re
import urllib.parse
from typing import Any, Dict, List, Literal, Optional

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURACIÓN DE ENTORNO Y CLIENTES
# ==============================================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
USE_WEBHOOK = os.getenv("USE_WEBHOOK", "false").strip().lower() in ("true", "1", "yes")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip().rstrip("/")
PORT = int(os.getenv("PORT", "10000"))

if not TELEGRAM_TOKEN:
    raise RuntimeError("Falta la variable de entorno TELEGRAM_TOKEN.")
if not GEMINI_API_KEY:
    raise RuntimeError("Falta la variable de entorno GEMINI_API_KEY.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

MODELOS_GEMINI = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# ==============================================================================
# ESQUEMA PYDANTIC
# ==============================================================================
class AnalisisConsultaEstudiante(BaseModel):
    intencion: Literal["reglamento", "clima", "certificado", "general"] = Field(
        description="Categoría principal del mensaje del estudiante."
    )
    ciudad_clima: Optional[str] = Field(
        default="Mendoza",
        description="Ciudad mencionada si la intención es consultar el clima.",
    )
    requiere_rag: bool = Field(
        default=False,
        description="Indica si conviene consultar el reglamento o normativa del ITU.",
    )
    resumen_consulta: str = Field(
        description="Resumen breve de lo que necesita el estudiante."
    )


# ==============================================================================
# BASE DE CONOCIMIENTO / RAG DEL REGLAMENTO ITU
# ==============================================================================
REGLAMENTO_FALLBACK = """
REGLAMENTO ACADÉMICO ITU - RESUMEN GENERAL:
1. Regularidad: En asignaturas presenciales se exige como mínimo un 75% de asistencia a clases teórico-prácticas y cumplimiento de las instancias evaluativas previstas en el programa. En modalidades virtuales o híbridas se evalúa la participación y entrega de trabajos prácticos semanales.
2. Promoción y exámenes: Cada cátedra establece los requisitos de aprobación directa o examen final según su programa vigente y el régimen académico del ITU.
3. Correlatividades: Para cursar o rendir mesas de examen de materias correlativas es obligatorio contar con la condición requerida (regularidad o aprobación) en las materias previas.
4. Certificado de alumno regular: Puede gestionarse mediante el trámite correspondiente indicando DNI y año de cursado actual.
""".strip()


def cargar_texto_reglamento() -> str:
    ruta_candidata = os.path.join(os.path.dirname(__file__), "reglamento_itu.txt")
    if os.path.exists(ruta_candidata):
        with open(ruta_candidata, "r", encoding="utf-8") as f:
            contenido = f.read().strip()
            if contenido:
                return contenido
    return REGLAMENTO_FALLBACK


TEXTO_REGLAMENTO = cargar_texto_reglamento()
CHUNKS_REGLAMENTO = [
    bloque.strip() for bloque in TEXTO_REGLAMENTO.split("\n\n") if bloque.strip()
] or [TEXTO_REGLAMENTO]


def recuperar_contexto_rag(consulta: str, top_k: int = 3) -> str:
    palabras = {
        p.lower()
        for p in re.findall(r"[a-záéíóúñ0-9]{3,}", consulta.lower())
    }
    if not palabras:
        return "\n\n".join(CHUNKS_REGLAMENTO[:top_k])

    puntuados = []
    for idx, chunk in enumerate(CHUNKS_REGLAMENTO):
        texto_chunk = chunk.lower()
        score = sum(1 for palabra in palabras if palabra in texto_chunk)
        puntuados.append((score, idx, chunk))

    puntuados.sort(key=lambda item: (-item[0], item[1]))
    mejores = [chunk for score, _, chunk in puntuados[:top_k] if score > 0]
    if not mejores:
        mejores = CHUNKS_REGLAMENTO[:top_k]
    return "\n\n".join(mejores)


# ==============================================================================
# HERRAMIENTA EXTERNA: CLIMA EN GRADOS CELSIUS (°C)
# ==============================================================================
def _fahrenheit_a_celsius(texto: str) -> str:
    """Convierte cualquier temperatura en °F (ej. +81°F o 81°F) a °C."""
    def _reemplazar(match: re.Match) -> str:
        valor_f = float(match.group(1))
        valor_c = round((valor_f - 32) * 5 / 9)
        signo = "+" if valor_c > 0 else ""
        return f"{signo}{valor_c}°C"

    return re.sub(r"([+-]?\d+(?:\.\d+)?)\s*°?F\b", _reemplazar, texto)


def consultar_clima(ciudad: str = "Mendoza") -> str:
    """Consulta el clima actual en una ciudad específica en grados Celsius (°C)."""
    ciudad_limpia = (ciudad or "Mendoza").strip()
    try:
        logger.info(f"-> [TOOL CALL] Ejecutando consultar_clima para: {ciudad_limpia}")
        ciudad_encoded = urllib.parse.quote(ciudad_limpia)
        url = f"https://wttr.in/{ciudad_encoded}?format=3&m"
        response = requests.get(url, timeout=8)
        if response.status_code == 200 and response.text.strip():
            return _fahrenheit_a_celsius(response.text.strip())
        return "No se pudo obtener el clima en este momento."
    except Exception as e:
        logger.error(f"Error en consultar_clima: {str(e)}")
        return f"Ocurrió un error al conectar con el servicio de clima: {str(e)}"


# ==============================================================================
# MEMORIA CONVERSACIONAL Y ESTADO DE TRÁMITE
# ==============================================================================
HISTORIAL_POR_USUARIO: Dict[int, List[Dict[str, str]]] = {}
ESTADO_TRAMITE: Dict[int, Dict[str, Any]] = {}
MAX_TURNOS_HISTORIAL = 8


def agregar_al_historial(user_id: int, rol: str, contenido: str) -> None:
    historial = HISTORIAL_POR_USUARIO.setdefault(user_id, [])
    historial.append({"rol": rol, "contenido": contenido})
    if len(historial) > MAX_TURNOS_HISTORIAL * 2:
        HISTORIAL_POR_USUARIO[user_id] = historial[-(MAX_TURNOS_HISTORIAL * 2) :]


def formatear_historial(user_id: int) -> str:
    historial = HISTORIAL_POR_USUARIO.get(user_id, [])
    if not historial:
        return "Sin historial previo."
    lineas = []
    for item in historial[-10:]:
        etiqueta = "Estudiante" if item["rol"] == "user" else "Samir"
        lineas.append(f"{etiqueta}: {item['contenido']}")
    return "\n".join(lineas)


# ==============================================================================
# LÓGICA CON GEMINI (PYDANTIC + RESPUESTA)
# ==============================================================================
def _llamar_gemini_con_fallback(prompt: str, config: types.GenerateContentConfig) -> str:
    ultimo_error: Optional[Exception] = None
    for modelo in MODELOS_GEMINI:
        try:
            resp = gemini_client.models.generate_content(
                model=modelo,
                contents=prompt,
                config=config,
            )
            if resp.text and resp.text.strip():
                return resp.text.strip()
        except Exception as e:
            ultimo_error = e
            logger.warning(f"Fallo con modelo {modelo}: {e}")
    raise RuntimeError(f"No se obtuvo respuesta de Gemini: {ultimo_error}")


def analizar_consulta_pydantic(mensaje: str, historial_texto: str) -> AnalisisConsultaEstudiante:
    prompt = f"""
Analiza el mensaje del estudiante del ITU y devuelve un objeto JSON válido según el esquema solicitado.
Si el mensaje pide clima, tiempo o temperatura, pon intencion="clima" y extrae ciudad_clima (si no menciona ciudad, usa "Mendoza").
Si pregunta por normas, regularidad, asistencia, promociones, correlatividades o mesas, pon intencion="reglamento" y requiere_rag=True.
Si pide certificado de alumno regular, pon intencion="certificado".

Historial reciente:
{historial_texto}

Mensaje actual:
{mensaje}
""".strip()

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=AnalisisConsultaEstudiante,
        temperature=0.1,
    )
    try:
        raw_json = _llamar_gemini_con_fallback(prompt, config)
        return AnalisisConsultaEstudiante.model_validate_json(raw_json)
    except Exception as e:
        logger.warning(f"Fallback heurístico de análisis para '{mensaje}': {e}")
        texto = mensaje.lower()
        if "clima" in texto or "temperatura" in texto or "c°" in texto or "°c" in texto:
            return AnalisisConsultaEstudiante(
                intencion="clima",
                ciudad_clima="Mendoza",
                requiere_rag=False,
                resumen_consulta=mensaje,
            )
        if "certificado" in texto:
            return AnalisisConsultaEstudiante(
                intencion="certificado",
                requiere_rag=False,
                resumen_consulta=mensaje,
            )
        return AnalisisConsultaEstudiante(
            intencion="reglamento",
            requiere_rag=True,
            resumen_consulta=mensaje,
        )


def generar_respuesta_samir(
    nombre_usuario: str,
    mensaje: str,
    analisis: AnalisisConsultaEstudiante,
    historial_texto: str,
    contexto_rag: str = "",
    dato_clima: str = "",
) -> str:
    instruccion_sistema = (
        "Eres Samir, el asistente virtual del ITU (Instituto Tecnológico Universitario - UNCUYO) "
        "para estudiantes de la Tecnicatura Universitaria en Desarrollo de Software. "
        "Responde en español, con tono cercano, claro y útil. "
        "Usa el historial reciente para mantener coherencia y no repitas saludos largos innecesarios "
        "si la conversación ya empezó."
    )

    extra_contexto = []
    if contexto_rag:
        extra_contexto.append(f"Información del reglamento ITU:\n{contexto_rag}")
    if dato_clima:
        extra_contexto.append(f"Dato meteorológico obtenido en °C:\n{dato_clima}")

    bloque_extra = "\n\n".join(extra_contexto) if extra_contexto else "Sin datos adicionales."

    prompt = f"""
Nombre del estudiante: {nombre_usuario}

Historial de conversación:
{historial_texto}

Análisis de la consulta:
{analisis.model_dump_json(indent=2)}

Contexto disponible:
{bloque_extra}

Consulta actual del estudiante:
{mensaje}
""".strip()

    config = types.GenerateContentConfig(
        system_instruction=instruccion_sistema,
        temperature=0.4,
    )
    return _llamar_gemini_con_fallback(prompt, config)


# ==============================================================================
# HANDLERS DE TELEGRAM
# ==============================================================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    user_id = update.effective_user.id
    nombre = update.effective_user.first_name or "estudiante"
    ESTADO_TRAMITE.pop(user_id, None)

    saludo = (
        f"¡Hola {nombre}! Soy Samir, tu asistente virtual del ITU. "
        f"¿En qué te puedo ayudar hoy con tu cursado o desarrollo de software?\n\n"
        f"💡 Comandos disponibles: /certificado (para iniciar el trámite de alumno regular)."
    )
    agregar_al_historial(user_id, "user", "/start")
    agregar_al_historial(user_id, "assistant", saludo)
    await update.message.reply_text(saludo)


async def cmd_certificado(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    user_id = update.effective_user.id
    ESTADO_TRAMITE[user_id] = {"paso": "dni", "datos": {}}

    mensaje = (
        "📄 *Iniciando trámite guiado:* Solicitud de Certificado de Alumno Regular.\n\n"
        "Por favor, ingresa tu *DNI* (solo números) para continuar, "
        "o escribe *cancelar* en cualquier momento para abortar el trámite."
    )
    agregar_al_historial(user_id, "user", "/certificado")
    agregar_al_historial(user_id, "assistant", mensaje)
    await update.message.reply_text(mensaje, parse_mode="Markdown")


async def procesar_paso_certificado(update: Update, user_id: int, texto: str) -> bool:
    tramite = ESTADO_TRAMITE.get(user_id)
    if not tramite:
        return False

    if not update.message:
        return False

    if texto.strip().lower() == "cancelar":
        ESTADO_TRAMITE.pop(user_id, None)
        resp = "❌ Trámite de certificado cancelado. Si necesitas retomarlo, envía /certificado."
        agregar_al_historial(user_id, "user", texto)
        agregar_al_historial(user_id, "assistant", resp)
        await update.message.reply_text(resp)
        return True

    paso = tramite["paso"]
    if paso == "dni":
        dni_limpio = re.sub(r"\D", "", texto)
        if len(dni_limpio) < 7:
            resp = "Por favor, ingresa un *DNI* válido (solo números, al menos 7 dígitos) o escribe *cancelar*."
            await update.message.reply_text(resp, parse_mode="Markdown")
            return True
        tramite["datos"]["dni"] = dni_limpio
        tramite["paso"] = "anio"
        resp = (
            "¡Perfecto! DNI registrado.\n\n"
            "Ahora, por favor, indica tu *año de cursado actual* (ej. 1er año, 2do año, 3er año):"
        )
        agregar_al_historial(user_id, "user", texto)
        agregar_al_historial(user_id, "assistant", resp)
        await update.message.reply_text(resp, parse_mode="Markdown")
        return True

    if paso == "anio":
        dni = tramite["datos"].get("dni", "")
        anio = texto.strip()
        ESTADO_TRAMITE.pop(user_id, None)
        resp = (
            "✅ *¡Trámite completado con éxito!*\n\n"
            "Hemos registrado tu solicitud de Certificado de Alumno Regular:\n"
            f"- *DNI:* {dni}\n"
            f"- *Año de cursado:* {anio}\n\n"
            "El certificado será enviado a tu correo institucional en un plazo de 24 a 48 horas hábiles. "
            "¡Mucho éxito con el cursado!"
        )
        agregar_al_historial(user_id, "user", texto)
        agregar_al_historial(user_id, "assistant", resp)
        await update.message.reply_text(resp, parse_mode="Markdown")
        return True

    return False


async def manejar_mensaje(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text or not update.effective_user:
        return

    user_id = update.effective_user.id
    nombre = update.effective_user.first_name or "estudiante"
    texto = update.message.text.strip()

    if await procesar_paso_certificado(update, user_id, texto):
        return

    try:
        historial_texto = formatear_historial(user_id)
        analisis = await asyncio.to_thread(analizar_consulta_pydantic, texto, historial_texto)

        dato_clima = ""
        contexto_rag = ""

        if analisis.intencion == "clima":
            ciudad = analisis.ciudad_clima or "Mendoza"
            clima_raw = await asyncio.to_thread(consultar_clima, ciudad)
            dato_clima = clima_raw
            respuesta = (
                f"🌤️ Información meteorológica para el ITU:\n"
                f"{clima_raw}"
            )
            agregar_al_historial(user_id, "user", texto)
            agregar_al_historial(user_id, "assistant", f"Clima consultado ({ciudad}): {clima_raw}")
            await update.message.reply_text(respuesta)
            return

        if analisis.intencion == "certificado":
            await cmd_certificado(update, context)
            return

        if analisis.requiere_rag or analisis.intencion == "reglamento":
            contexto_rag = recuperar_contexto_rag(texto)

        respuesta = await asyncio.to_thread(
            generar_respuesta_samir,
            nombre,
            texto,
            analisis,
            historial_texto,
            contexto_rag,
            dato_clima,
        )

        agregar_al_historial(user_id, "user", texto)
        agregar_al_historial(user_id, "assistant", respuesta)
        await update.message.reply_text(respuesta)

    except Exception as e:
        logger.error(f"Error procesando mensaje de {user_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "Disculpa, tuve un pequeño inconveniente procesando tu solicitud. ¡Inténtalo de nuevo!"
        )


# ==============================================================================
# ARRANQUE (WEBHOOK EN RENDER / POLLING EN LOCAL)
# ==============================================================================
def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("certificado", cmd_certificado))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manejar_mensaje))

    if USE_WEBHOOK and WEBHOOK_URL:
        url_path = TELEGRAM_TOKEN
        webhook_full_url = f"{WEBHOOK_URL}/{url_path}"
        logger.info(f"¡Bot Samir iniciado en modo WEBHOOK en el puerto {PORT}!")
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=url_path,
            webhook_url=webhook_full_url,
            drop_pending_updates=True,
        )
    else:
        logger.info("¡Bot Samir iniciado en modo POLLING!")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()