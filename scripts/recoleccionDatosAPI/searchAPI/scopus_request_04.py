import requests
import os
from datetime import datetime, timezone

from scripts.recoleccionDatosAPI.searchAPI.config_00 import (
    BASE_URL, HEADERS, CONSULTAS_PATH, BASE_DIR,
    sleep_entre_llamadas, sleep_minimo, sleep_maximo,
    consultas_pausa, consultas_gran_pausa, PAGE_SIZE
)
from scripts.recoleccionDatosAPI.searchAPI.logger_01 import log_info, log_warning, log_debug
from scripts.recoleccionDatosAPI.searchAPI.utils_02 import esperar_segundos
from scripts.recoleccionDatosAPI.searchAPI.data_io_03 import log_limite_remaining, backup_contador
from urllib.parse import quote

# === VARIABLES DE ESTADO ===
errores_recientes = 0
errores_limite = 3
exitos_recientes = 0
consultas_realizadas = 0
consultas_realizadas_total = 0

# === FUNCIÓN DE CONSULTA ===
def hacer_request(params):
    global errores_recientes, exitos_recientes
    global consultas_realizadas, consultas_realizadas_total
    global sleep_entre_llamadas

    espera = sleep_entre_llamadas

    for intento in range(5):
        try:
            response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=20)
            log_limite_remaining(response.headers)

            if response.status_code == 429:
                log_warning("🔄 429 Too Many Requests detectado.")
                if intento < 4:
                    espera_expo = min(60 * (intento + 1), 600)  # 60s, 120s, 180s... máx 600s
                    log_info(f"⏳ Esperando {espera_expo//60}m {espera_expo%60}s antes de reintentar...")
                    esperar_segundos(espera_expo)
                else:
                    esperar_por_limite(response)
                continue

            response.raise_for_status()

            errores_recientes = max(errores_recientes - 1, 0)
            exitos_recientes += 1
            consultas_realizadas += 1
            consultas_realizadas_total += 1

            if exitos_recientes >= 50 and sleep_entre_llamadas > sleep_minimo:
                sleep_entre_llamadas = max(sleep_minimo, sleep_entre_llamadas - 0.1)
                exitos_recientes = 0
                log_info(f"🚀 Acelerando! Nuevo sleep: {sleep_entre_llamadas:.2f}s")

            if consultas_realizadas >= consultas_pausa:
                log_info("☕ Pausa corta: 5 minutos tras consultas.")
                esperar_segundos(300)
                consultas_realizadas = 0

            if consultas_realizadas_total >= consultas_gran_pausa:
                log_info("🛑 Pausa larga: 60 minutos tras muchas consultas.")
                backup_contador()
                try:
                    esperar_segundos(3600)
                except KeyboardInterrupt:
                    log_info("🛑 Interrupción manual durante pausa larga. Saliendo del proceso.")
                    raise
                consultas_realizadas = 0
                consultas_realizadas_total = 0

            return response.json()

        except requests.exceptions.RequestException as e:
            errores_recientes += 1
            exitos_recientes = 0
            log_warning(f"⚠️ Error de conexión: {e}. Reintentando...")
            esperar_segundos(espera)
            espera = min(espera * 2, sleep_maximo)

            if errores_recientes >= errores_limite:
                sleep_entre_llamadas = min(sleep_entre_llamadas * 2, sleep_maximo)
                errores_recientes = 0
                log_info(f"🐢 Aumentando espera a {sleep_entre_llamadas:.2f}s...")

    raise Exception("❌ Fallo tras múltiples reintentos.")

# === FUNCIÓN DE ESPERA POR CABECERA ===
def esperar_por_limite(response):
    reset_timestamp = response.headers.get("X-RateLimit-Reset")
    if reset_timestamp:
        try:
            reset_time = datetime.fromtimestamp(int(reset_timestamp), tz=timezone.utc)
            now = datetime.now(timezone.utc)
            wait_seconds = (reset_time - now).total_seconds()
            if wait_seconds > 0:
                log_info(f"⏳ Esperando {int(wait_seconds // 60)} min {int(wait_seconds % 60)} seg hasta desbloqueo...")
                esperar_segundos(wait_seconds + 2)
        except Exception as e:
            log_warning(f"⏳ Error procesando cabecera X-RateLimit-Reset: {e}. Esperando 60s...")
            esperar_segundos(60)
    else:
        log_info("⏳ Sin cabecera X-RateLimit-Reset. Esperando 60s...")
        esperar_segundos(60)

def hacer_request_cursor(query_base, cursor):
    global errores_recientes, exitos_recientes
    global consultas_realizadas, consultas_realizadas_total
    global sleep_entre_llamadas

    espera = sleep_entre_llamadas
    query_encoded = quote(query_base)

    for intento in range(5):
        try:
            params = {
                "query": query_base,
                "cursor": cursor,
                "count": PAGE_SIZE,
                "view": "STANDARD"
            }
            url_debug = requests.Request(
                "GET", BASE_URL, params={
                    "query": query_base,
                    "cursor": cursor,
                    "count": PAGE_SIZE,
                    "view": "STANDARD"
                }
            ).prepare().url
            log_debug(f"🔗 URL solicitada: {url_debug}")
            response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=20)
            log_limite_remaining(response.headers)

            if response.status_code == 429:
                log_warning("🔄 429 Too Many Requests detectado.")
                if intento < 4:
                    espera_expo = min(60 * (intento + 1), 600)  # 60s, 120s, 180s... máx 600s
                    log_info(f"⏳ Esperando {espera_expo//60}m {espera_expo%60}s antes de reintentar...")
                    esperar_segundos(espera_expo)
                else:
                    esperar_por_limite(response)
                continue

            response.raise_for_status()

            errores_recientes = max(errores_recientes - 1, 0)
            exitos_recientes += 1
            consultas_realizadas += 1
            consultas_realizadas_total += 1

            if exitos_recientes >= 50 and sleep_entre_llamadas > sleep_minimo:
                sleep_entre_llamadas = max(sleep_minimo, sleep_entre_llamadas - 0.1)
                exitos_recientes = 0
                log_info(f"🚀 Acelerando (cursor)! Nuevo sleep: {sleep_entre_llamadas:.2f}s")

            if consultas_realizadas >= consultas_pausa:
                log_info("☕ Pausa corta (cursor): 5 minutos tras consultas.")
                esperar_segundos(300)
                consultas_realizadas = 0

            if consultas_realizadas_total >= consultas_gran_pausa:
                log_info("🛑 Pausa larga (cursor): 60 minutos tras muchas consultas.")
                backup_contador()
                try:
                    esperar_segundos(3600)
                except KeyboardInterrupt:
                    log_info("🛑 Interrupción manual durante pausa larga. Saliendo del proceso.")
                    log_info("🔁 Interrupción ignorada. Continuando sin pausar más.")
                    break
                consultas_realizadas = 0
                consultas_realizadas_total = 0

            return response.json()

        except requests.exceptions.RequestException as e:
            errores_recientes += 1
            exitos_recientes = 0
            log_warning(f"⚠️ Error de conexión (cursor): {e}. Reintentando...")
            esperar_segundos(espera)
            espera = min(espera * 2, sleep_maximo)

            if errores_recientes >= errores_limite:
                sleep_entre_llamadas = min(sleep_entre_llamadas * 2, sleep_maximo)
                errores_recientes = 0
                log_info(f"🐢 Aumentando espera (cursor) a {sleep_entre_llamadas:.2f}s...")

    raise Exception("❌ Fallo tras múltiples reintentos (cursor).")

def guardar_cursor(anio, cursor_valor):
    ruta = os.path.join(BASE_DIR, "data", "tmp_cursor", f"cursor_{anio}.txt")
    log_debug(f"💾 Intentando guardar cursor en: {ruta}")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(cursor_valor.strip())

def cargar_cursor(anio):
    ruta = os.path.join(BASE_DIR, "data", "tmp_cursor", f"cursor_{anio}.txt")
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "*"

def eliminar_cursor(anio):
    ruta = os.path.join(BASE_DIR, "data", "tmp_cursor", f"cursor_{anio}.txt")
    if os.path.exists(ruta):
        os.remove(ruta)
