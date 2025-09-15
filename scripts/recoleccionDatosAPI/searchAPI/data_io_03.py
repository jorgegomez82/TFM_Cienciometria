import os
import csv
import json
from typing import List, Dict, Set
from datetime import datetime, timezone

from scripts.recoleccionDatosAPI.searchAPI.config_00 import BASE_DIR, LOG_PATH, CONSULTAS_PATH, PROGRESO_PATH, DIR_RAW, LIMITE_QUOTA_PATH
from scripts.recoleccionDatosAPI.searchAPI.utils_02 import limpiar_texto
from scripts.recoleccionDatosAPI.searchAPI.logger_01 import log_info, log_warning

# === REGISTRO DEL LÍMITE DE USO DE LA API ===
def log_limite_remaining(headers: dict, base_dir: str = None) -> None:
    remaining = headers.get("X-RateLimit-Remaining")
    reset = headers.get("X-RateLimit-Reset")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if remaining:
        try:
            remaining_int = int(remaining)
            if remaining_int < 5000:
                with open(LIMITE_QUOTA_PATH, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp} - X-RateLimit-Remaining: {remaining_int}\n")
        except ValueError:
            pass

    if reset:
        try:
            reset_time = datetime.fromtimestamp(int(reset), tz=timezone.utc)
            now = datetime.now(timezone.utc)
            segundos_restantes = (reset_time - now).total_seconds()
            if segundos_restantes < 600:
                backup_contador()
        except Exception as e:
            log_warning(f"⚠️ Error al procesar X-RateLimit-Reset: {e}")

# === BACKUP DEL CONTADOR DE CONSULTAS ===
def backup_contador() -> None:
    valor = cargar_contador_consultas()
    nombre = f"consultas_backup_reset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    ruta = os.path.join(DIR_RAW, nombre)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(str(valor))
    log_info(f"🔒 Backup de contador guardado en {ruta}")

# === FUNCIONES DE GESTIÓN DE AÑOS Y CONTADOR ===
def cargar_anios_procesados() -> Set[int]:
    if not os.path.exists(LOG_PATH):
        return set()
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())

def guardar_anio_procesado(anio: int) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{anio}\n")

def cargar_contador_consultas() -> int:
    if not os.path.exists(CONSULTAS_PATH):
        return 0
    with open(CONSULTAS_PATH, "r", encoding="utf-8") as f:
        return int(f.read().strip() or 0)

def guardar_contador_consultas(valor: int) -> None:
    with open(CONSULTAS_PATH, "w", encoding="utf-8") as f:
        f.write(str(valor))

# === GUARDADO DE ARTÍCULOS ===
def nombre_archivo_por_decada(anio: int) -> str:
    decada_inicio = (anio // 10) * 10
    return os.path.join(DIR_RAW, f"articulos_scopus_{decada_inicio}_{decada_inicio + 9}.csv")

def guardar_articulos(articulos: List[Dict[str, str]], anio: int) -> int:
    archivo = nombre_archivo_por_decada(anio)
    existe = os.path.isfile(archivo)
    eids_existentes = set()

    if existe:
        with open(archivo, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            eids_existentes = {row["eid"] for row in reader if "eid" in row and row["eid"]}

    nuevos_articulos = []
    for art in articulos:
        eid = art.get("eid", "").strip()
        if eid and eid not in eids_existentes:
            articulo_limpio = {clave: limpiar_texto(valor) for clave, valor in art.items()}
            nuevos_articulos.append(articulo_limpio)

    log_info(f"📥 Nuevos artículos a guardar para {anio}: {len(nuevos_articulos)}")

    if nuevos_articulos:
        with open(archivo, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=nuevos_articulos[0].keys())
            if not existe:
                writer.writeheader()
            writer.writerows(nuevos_articulos)

    return len(nuevos_articulos)

# === REGISTRO DE PROGRESO ===
def registrar_progreso(decada_inicio: int, decada_fin: int, duracion: float, descargados: int,
                       consultas: int, tiempo_estimado: float, restantes: int) -> None:
    def formato(segundos: float) -> str:
        h = int(segundos // 3600)
        m = int((segundos % 3600) // 60)
        s = int(segundos % 60)
        return f"{h}h {m}m {s}s"

    with open(PROGRESO_PATH, "a", encoding="utf-8") as f:
        f.write(f"Decada {decada_inicio}-{decada_fin} completada\n")
        f.write(f"Duracion acumulada: {formato(duracion)}\n")
        f.write(f"Articulos descargados: {descargados}\n")
        f.write(f"Consultas realizadas: {consultas}\n")
        f.write(f"Estimado restante: {formato(tiempo_estimado)}\n")
        f.write(f"Decadas restantes: {restantes}\n")
        f.write("-------------------------------\n")

# === TEMPORAL: GUARDADO Y CARGA DE ARTÍCULOS POR AÑO ===
def guardar_articulos_temporalmente(anio: int, articulos: List[dict]) -> None:
    """Guarda incrementalmente artículos en JSON por año."""
    ruta = os.path.join(BASE_DIR, "data/tmp_articulos", f"articulos_{anio}.json")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)

    existentes = []
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            try:
                existentes = json.load(f)
            except json.JSONDecodeError:
                existentes = []

    nuevos = [art for art in articulos if art.get("eid") and art["eid"] not in {x["eid"] for x in existentes if "eid" in x}]
    if nuevos:
        existentes += nuevos
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(existentes, f, indent=2, ensure_ascii=False)


def cargar_articulos_temporales(anio: int) -> List[dict]:
    """Carga artículos previamente almacenados en JSON temporal."""
    ruta = os.path.join(BASE_DIR, "data/tmp_articulos", f"articulos_{anio}.json")
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []