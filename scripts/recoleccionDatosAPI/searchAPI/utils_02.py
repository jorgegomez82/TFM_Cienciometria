# utils.py
import unicodedata
from datetime import datetime
from scripts.recoleccionDatosAPI.searchAPI.logger_01 import log_info, log_warning

def limpiar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return texto
    texto = unicodedata.normalize("NFKC", texto)
    texto = texto.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return texto

def formato_segundos(segundos: float) -> str:
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segundos = int(segundos % 60)
    return f"{horas}h {minutos}m {segundos}s"

def timestamp_actual() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def esperar_segundos(segundos):
    from time import sleep
    try:
        log_info(f"⏳ Esperando {formato_segundos(segundos)}...")
        sleep(segundos)
    except KeyboardInterrupt:
        log_warning("🛑 Espera interrumpida por el usuario.")
        raise
