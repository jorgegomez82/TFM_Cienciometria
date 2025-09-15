import logging
import os
from logging.handlers import RotatingFileHandler
from scripts.recoleccionDatosAPI.searchAPI.config_00 import BASE_DIR

# === CONFIGURAR DIRECTORIO Y ARCHIVO DE LOG ===
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "descarga_scopus.log")

# === CONFIGURAR LOGGING CON ROTACIÓN ===
log_file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log_file_handler,
        logging.StreamHandler()
    ]
)

# === FUNCIONES DE LOG ===
def log_info(mensaje):
    logging.info(mensaje)

def log_warning(mensaje):
    logging.warning(mensaje)

def log_error(mensaje):
    logging.error(mensaje)

def log_debug(mensaje):
    logging.debug(mensaje)
