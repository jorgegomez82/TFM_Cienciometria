# === config.py ===
import os
from dotenv import load_dotenv

# === CARGA DE VARIABLES ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
load_dotenv(os.path.join(BASE_DIR, "seguridad.env"))

API_KEY = os.getenv("SCOPUS_API_KEY")
INST_TOKEN = os.getenv("SCOPUS_INST_TOKEN")

if not API_KEY or not INST_TOKEN:
    raise ValueError("API_KEY o INST_TOKEN no están definidos en el archivo .env")

# === PARÁMETROS DE CONSULTA ===
START_YEAR, END_YEAR = (1826, 2024)
max_pages = 199
PAGE_SIZE = 25

# === CONFIGURACIÓN DE CURSOR PAGINATION ===
MAX_PAGINAS_VACIAS_CONSECUTIVAS = 3

# === TIEMPOS DE ESPERA Y LÍMITES ===
sleep_entre_llamadas = 1.0
sleep_minimo = 0.1
sleep_maximo = 10.0
consultas_pausa = 500
consultas_gran_pausa = 3000

# === RUTAS DE ARCHIVOS ===
DIR_RAW = os.path.join(BASE_DIR, "data", "raw")
LOG_PATH = os.path.join(BASE_DIR, "log_anios_procesados.txt")
PROGRESO_PATH = os.path.join(BASE_DIR, "progreso_descarga.txt")
CONSULTAS_PATH = os.path.join(BASE_DIR, "contador_consultas.txt")
LIMITE_QUOTA_PATH = os.path.join(BASE_DIR, "log_limite_quota.txt")

# === HEADERS Y ENDPOINT DE LA API ===
HEADERS = {
    "X-ELS-APIKey": API_KEY,
    "X-ELS-Insttoken": INST_TOKEN,
    "Accept": "application/json"
}

BASE_URL = "https://api.elsevier.com/content/search/scopus"