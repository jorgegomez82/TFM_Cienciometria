import os
import csv
import json
import time
import requests
import argparse
from datetime import datetime
from dotenv import load_dotenv

# === ARGUMENTOS ===
parser = argparse.ArgumentParser(description="Enriquecimiento de artículos desde Abstract Retrieval API")
parser.add_argument("--test", action="store_true", help="Ejecutar solo 2 artículos como prueba")
parser.add_argument("--solo_resumen", action="store_true", help="Mostrar solo el resumen semanal sin descargar")
args = parser.parse_args()

if args.solo_resumen:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    PROGRESO_PATH = os.path.join(BASE_DIR, "progreso_descarga.txt")
    REPORTE_SEMANAL_DIR = os.path.join(BASE_DIR, "logs", "reportes_semanales")
    os.makedirs(REPORTE_SEMANAL_DIR, exist_ok=True)

    from datetime import datetime, timedelta, timezone
    semana_id = datetime.now().strftime("%Y_%m_%d")
    total_articulos = 3640783
    progreso_total = 0
    if os.path.exists(PROGRESO_PATH):
        with open(PROGRESO_PATH, "r", encoding="utf-8") as f:
            progreso_total = len(set(f.read().splitlines()))
    porcentaje_total = round((progreso_total / total_articulos) * 100, 2)
    semanas_restantes = round((total_articulos - progreso_total) / 150000)

    now = datetime.now(timezone.utc)
    next_monday = now + timedelta(days=(7 - now.weekday()))
    next_reset = datetime.combine(next_monday.date(), datetime.min.time(), tzinfo=timezone.utc)
    time_remaining = next_reset - now
    horas, resto = divmod(time_remaining.total_seconds(), 3600)
    minutos = int(resto // 60)

    resumen_json = {
        "semana": semana_id,
        "progreso_total": progreso_total,
        "porcentaje_total": porcentaje_total,
        "estimacion_semanas_restantes": semanas_restantes,
        "horas_para_nueva_cuota": int(horas),
        "minutos_para_nueva_cuota": minutos
    }

    reporte_path = os.path.join(REPORTE_SEMANAL_DIR, f"reporte_semanal_{semana_id}.json")
    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(resumen_json, f, ensure_ascii=False, indent=2)

    print("\n=== RESUMEN SEMANAL ===")
    for k, v in resumen_json.items():
        print(f"{k}: {v}")
    exit()

# === CARGA DE VARIABLES DE ENTORNO ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
load_dotenv(os.path.join(BASE_DIR, "seguridad.env"))

API_KEY = os.getenv("SCOPUS_API_KEY")
INST_TOKEN = os.getenv("SCOPUS_INST_TOKEN")

if not API_KEY or not INST_TOKEN:
    raise ValueError("SCOPUS_API_KEY o SCOPUS_INST_TOKEN no están definidos en el archivo .env")

HEADERS = {
    "Accept": "application/json",
    "X-ELS-APIKey": API_KEY,
    "X-ELS-Insttoken": INST_TOKEN,
}

API_URL_TEMPLATE = "https://api.elsevier.com/content/abstract/eid/{}?view=FULL"

CSV_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "tmp_articulos_abstract")
ENRIQUECIDO_DIR = os.path.join(BASE_DIR, "data", "enriquecido")
LOG_PATH = os.path.join(BASE_DIR, "log_abstract_anios_procesados.txt")
ERROR_LOG = os.path.join(BASE_DIR, "logs", "descarga_errores.log")
PROGRESO_PATH = os.path.join(BASE_DIR, "progreso_descarga.txt")
RESUMEN_LOG = os.path.join(BASE_DIR, "logs", "resumen_descarga.log")
REPORTE_SEMANAL_DIR = os.path.join(BASE_DIR, "logs", "reportes_semanales")
EIDS_EXCLUIDOS_PATH = os.path.join(BASE_DIR, "logs", "eids_irrecuperables.txt")
OMITIDOS_LOG_PATH = os.path.join(BASE_DIR, "logs", "omitidos_por_exclusion.log")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ENRIQUECIDO_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ERROR_LOG), exist_ok=True)
os.makedirs(REPORTE_SEMANAL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(EIDS_EXCLUIDOS_PATH), exist_ok=True)

def solicitud_con_reintentos(url, headers, intentos=5):
    consecutivos_429 = 0

    for intento in range(intentos):
        try:
            resp = requests.get(url, headers=headers, timeout=(10, 60))

            limit = resp.headers.get("X-RateLimit-Limit")
            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset = resp.headers.get("X-RateLimit-Reset")

            if remaining is not None and limit is not None:
                print(f"\n📊 Cuota: {remaining} / {limit} consultas restantes")

                if int(remaining) == 0:
                    wait_seconds = 300
                    if reset:
                        now = datetime.now(timezone.utc)
                        wait_seconds = int(reset) - int(now.timestamp())
                        wait_seconds = max(wait_seconds, 60)
                    print(f"⛔ Consultas restantes = 0. Esperando {wait_seconds} segundos hasta reinicio de cuota...")
                    time.sleep(wait_seconds)
                    continue

            if resp.status_code == 429:
                consecutivos_429 += 1
                print(f"🚫 Código 429 recibido para {url}")
                wait_seconds = 300
                if reset:
                    now = datetime.now(timezone.utc)
                    wait_seconds = int(reset) - int(now.timestamp())
                    wait_seconds = max(wait_seconds, 60)
                else:
                    print("⚠️ X-RateLimit-Reset no disponible. Usando espera de 300s...")

                print(f"⏳ Esperando {wait_seconds} segundos... ({consecutivos_429}/{intentos})")
                time.sleep(wait_seconds)

                if consecutivos_429 >= intentos:
                    raise RuntimeError(f"🚫 Se recibieron {intentos} errores 429 consecutivos. Deteniendo para evitar bloqueo.")

                continue

            if resp.status_code == 404:
                raise RuntimeError(f"404 Not Found para {url}")
            resp.raise_for_status()
            return resp  # OK

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error en la solicitud ({intento + 1}/{intentos}) para {url}: {e}")
            time.sleep(min(2 ** intento, 60))

    raise RuntimeError(f"❌ No se pudo conectar con la API tras múltiples intentos. URL: {url}")

def cargar_progreso():
    if os.path.exists(PROGRESO_PATH):
        with open(PROGRESO_PATH, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f)
    return set()

def guardar_progreso(eid):
    with open(PROGRESO_PATH, "a", encoding="utf-8") as f:
        f.write(eid + "\n")

def cargar_eids_excluidos():
    if os.path.exists(EIDS_EXCLUIDOS_PATH):
        with open(EIDS_EXCLUIDOS_PATH, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f)
    return set()

def guardar_eid_excluido(eid):
    with open(EIDS_EXCLUIDOS_PATH, "a", encoding="utf-8") as f:
        f.write(eid + "\n")

def registrar_eid_omitido(eid):
    with open(OMITIDOS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(eid + "\n")

# Cargar EIDs excluidos al comienzo del script
eids_excluidos = cargar_eids_excluidos()

def registrar_error(eid, status):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"{eid},{status}\n")

def registrar_error_estructura(eid, mensaje):
    path = os.path.join(BASE_DIR, "logs", "errores_json_estructurales.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{eid},{mensaje}\n")

def esta_procesado(csv_file):
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return csv_file in set(line.strip() for line in f)
    return False

def marcar_como_procesado(csv_file):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(csv_file + "\n")

def extraer_campos(respuesta, eid_original):
    try:
        item = respuesta.get("abstracts-retrieval-response")
        if not item:
            raise ValueError("Respuesta sin 'abstracts-retrieval-response'")

        affiliations = item.get("affiliation", [])
        if not isinstance(affiliations, list):
            affiliations = []

        return {
            "eid": eid_original,
            "dc:description": item.get("coredata", {}).get("dc:description"),
            "citedby-count": item.get("coredata", {}).get("citedby-count"),
            "affiliations": affiliations,
            "authors": item.get("authors", {}).get("author", []),
            "keywords": item.get("item", {}).get("bibrecord", {}).get("head", {}).get("citation-info", {}).get("author-keywords", {}).get("author-keyword", []),
            "subject-areas": item.get("subject-areas", {}),
            "item": item.get("item", {})  # necesario para el idioma
        }
    except Exception as e:
        registrar_error_estructura(eid_original, str(e))
        return {"eid": eid_original, "error": str(e)}

def procesar_csv(csv_path, eids_descargados, resumen, limite_descargas):
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        total_filas = len(reader)
        descargas_anteriores = resumen["descargados"]
        contador_local = 0

        for i, row in enumerate(reader, 1):
            print(" " * 100, end="\r")  # limpiar línea anterior
            print(f"\r📥 Procesando artículo {i} / {total_filas} del archivo {os.path.basename(csv_path)}", end="", flush=True)

            if resumen["descargados"] >= limite_descargas:
                break

            eid = row["eid"]
            if eid in eids_descargados:
                continue
            if eid in eids_excluidos:
                print(f"⚠️ EID {eid} omitido por estar marcado como irrecuperable.")
                registrar_eid_omitido(eid)
                continue
           

            out_file = os.path.join(OUT_DIR, f"metadata_{eid}.json")
            if os.path.exists(out_file):
                guardar_progreso(eid)
                eids_descargados.add(eid)
                resumen["existentes"] += 1
                continue

            url = API_URL_TEMPLATE.format(eid)
            try:
                resp = solicitud_con_reintentos(url, HEADERS)
                if resp.status_code == 200:
                    data = extraer_campos(resp.json(), eid_original=eid)
                    with open(out_file, "w", encoding="utf-8") as f_out:
                        json.dump(data, f_out, ensure_ascii=False, indent=2)
                    guardar_progreso(eid)
                    eids_descargados.add(eid)
                    resumen["descargados"] += 1
                    contador_local += 1

                else:
                    registrar_error(eid, resp.status_code)
                    resumen["errores"] += 1
            except Exception as e:
                registrar_error(eid, str(e))
                guardar_eid_excluido(eid)
                resumen["errores"] += 1

            # 💾 Guardado parcial cada 500 artículos nuevos
            if contador_local > 0 and contador_local % 500 == 0:
                print(f"\n💾 Guardado parcial tras {contador_local} artículos descargados en {os.path.basename(csv_path)}")
                fusionar_datos(csv_path, eids_descargados)

            time.sleep(1 / 9)

def guardar_resumen(resumen):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    resumen_texto = (
        f"=== RESUMEN DE DESCARGA ({timestamp}) ===\n"
        f"Artículos descargados nuevos: {resumen['descargados']}\n"
        f"Artículos ya existentes: {resumen['existentes']}\n"
        f"Errores durante la descarga: {resumen['errores']}\n"
    )
    print("\n" + resumen_texto)
    with open(RESUMEN_LOG, "a", encoding="utf-8") as f:
        f.write(resumen_texto + "\n")

    semana_id = datetime.now().strftime("%Y_%m_%d")
    reporte_path = os.path.join(REPORTE_SEMANAL_DIR, f"reporte_semanal_{semana_id}.json")
    total_articulos = 3640783
    progreso_total = len(cargar_progreso())
    porcentaje_total = round((progreso_total / total_articulos) * 100, 2)
    semanas_restantes = round((total_articulos - progreso_total) / 150000)

    resumen_json = {
        "semana": semana_id,
        "descargados_nuevos": resumen['descargados'],
        "existentes": resumen['existentes'],
        "errores": resumen['errores'],
        "progreso_total": progreso_total,
        "porcentaje_total": porcentaje_total,
        "estimacion_semanas_restantes": semanas_restantes
    }

    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(resumen_json, f, ensure_ascii=False, indent=2)

def fusionar_datos(csv_path, eids_descargados):
    enriched_file = os.path.join(ENRIQUECIDO_DIR, f"enriquecido_{os.path.basename(csv_path)}")
    existentes = set()
    if os.path.exists(enriched_file):
        with open(enriched_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existentes.add(row["eid"])

    nuevos_rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row["eid"]
            if eid in existentes:
                continue  # ya fue fusionado previamente

            json_path = os.path.join(OUT_DIR, f"metadata_{eid}.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path, "r", encoding="utf-8") as jf:
                enriched = json.load(jf)

            if "error" in enriched:
                continue

            autores = []
            autores_ids = []
            for a in enriched.get("authors", []):
                name = a.get("preferred-name", {}).get("ce:indexed-name")
                if name:
                    autores.append(name)
                aid = a.get("@auid")
                if aid:
                    autores_ids.append(str(aid))

            aff_data = enriched.get("affiliations", [])
            if not isinstance(aff_data, list):
                aff_data = []

            if not aff_data:
                for a in enriched.get("authors", []):
                    aff = a.get("affiliation")
                    if isinstance(aff, dict):
                        aff_data.append(aff)

            afiliaciones = [aff.get("affiliation-name", "") for aff in aff_data if isinstance(aff, dict)]
            afids = [aff.get("@id", "") for aff in aff_data if isinstance(aff, dict)]

            mapa_afiliaciones = {}
            for aff in aff_data:
                if isinstance(aff, dict):
                    clave = aff.get("@id")
                    valor = aff.get("affilname") or aff.get("affiliation-name")
                    if clave and valor:
                        mapa_afiliaciones[clave] = valor

            nombres_afiliaciones_extra = []
            for a in enriched.get("authors", []):
                aff = a.get("affiliation")
                if isinstance(aff, dict):
                    aid = aff.get("@id")
                    nombre = mapa_afiliaciones.get(aid)
                    if not nombre:
                        nombre = aff.get("affilname") or aff.get("affiliation-name") or "(sin nombre)"
                    nombres_afiliaciones_extra.append(nombre)

            afiliaciones = list(dict.fromkeys(nombres_afiliaciones_extra))
            afiliaciones = [a.strip() for a in afiliaciones if a.strip()]

            # === Agregar subject_areas, num_subdisciplinas, idioma ===
            subject_areas_obj = enriched.get("subject-areas", {})
            subject_areas_data = []

            if isinstance(subject_areas_obj, dict):
                sa = subject_areas_obj.get("subject-area", [])
                if isinstance(sa, dict):
                    subject_areas_data = [sa]
                elif isinstance(sa, list):
                    subject_areas_data = sa

            subdisciplinas_matematicas = []
            for s in subject_areas_data:
                if not isinstance(s, dict):
                    continue
                try:
                    code = int(s.get("@code", 0))
                    if 2602 <= code <= 2614:
                        nombre = s.get("$", "")
                        subdisciplinas_matematicas.append(f"{code}-{nombre}")
                except ValueError:
                    continue

            subjects_str = "; ".join(subdisciplinas_matematicas)
            num_subdisciplinas = len(subdisciplinas_matematicas)

            idioma_raw = enriched.get("item", {}) \
                .get("bibrecord", {}) \
                .get("head", {}) \
                .get("citation-info", {}) \
                .get("citation-language", {})

            if isinstance(idioma_raw, dict):
                idioma = idioma_raw.get("@xml:lang", "")
            elif isinstance(idioma_raw, list) and len(idioma_raw) > 0 and isinstance(idioma_raw[0], dict):
                idioma = idioma_raw[0].get("@xml:lang", "")
            else:
                idioma = ""

            raw_keywords = enriched.get("keywords", [])
            if isinstance(raw_keywords, dict):
                raw_keywords = [raw_keywords]
            elif not isinstance(raw_keywords, list):
                raw_keywords = []

            row.update({
                "abstract": enriched.get("dc:description"),
                "citedby_count": enriched.get("citedby-count"),                
                "keywords": "; ".join(k["$"] if isinstance(k, dict) and "$" in k else str(k) for k in raw_keywords),
                "num_authors": len(autores),
                "num_affiliations": len(afiliaciones),
                "authors_full": "; ".join(autores),
                "authors_ids": "; ".join(autores_ids),
                "affiliations_full": "; ".join(afiliaciones),
                "affiliations_ids": "; ".join(afids),
                "subject_areas": subjects_str,
                "num_subdisciplinas": num_subdisciplinas,
                "idioma": idioma
            })
            nuevos_rows.append(row)

            try:
                os.remove(json_path)
            except Exception as e:
                print(f"⚠️ No se pudo eliminar {json_path}: {e}")

    if nuevos_rows:
        with open(enriched_file, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=nuevos_rows[0].keys())
            if os.stat(enriched_file).st_size == 0:
                writer.writeheader()
            writer.writerows(nuevos_rows)
        print(f"\n✅ {len(nuevos_rows)} artículos nuevos fusionados en {os.path.basename(csv_path)}")

# === PROCESAMIENTO GENERAL ===
csv_files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith(".csv")])
eids_descargados = cargar_progreso()
resumen = {"descargados": 0, "errores": 0, "existentes": 0}

limite_total = 2 if args.test else 150000

try:
    for csv_file in csv_files:
        if resumen["descargados"] >= limite_total:
            break

        if esta_procesado(csv_file):
            print(f"✅ {csv_file} ya fue completamente procesado. Se omite.")
            continue

        csv_path = os.path.join(CSV_DIR, csv_file)
        procesar_csv(csv_path, eids_descargados, resumen, limite_total)

        if resumen["descargados"] >= limite_total:
            print(f"\n⏹️ Límite alcanzado antes de terminar {csv_file}. Se procederá a fusionar los artículos descargados hasta el momento.")
            try:
                fusionar_datos(csv_path, eids_descargados)
            except Exception as e:
                print(f"❌ Error al fusionar parcialmente {csv_file}: {e}")
            break

        try:
            fusionar_datos(csv_path, eids_descargados)
            marcar_como_procesado(csv_file)
        except Exception as e:
            print(f"\n❌ Error al fusionar {csv_file}: {e}")

except KeyboardInterrupt:
    print("\n🛑 Interrupción detectada con Ctrl + C. Guardando CSV enriquecido parcial...")

    # Fusionar el último CSV parcialmente descargado (si existe)
    if 'csv_path' in locals():
        fusionar_datos(csv_path, eids_descargados)

finally:
    guardar_resumen(resumen)