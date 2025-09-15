# procesador.py
from time import sleep
from tqdm import tqdm

from scripts.recoleccionDatosAPI.searchAPI.logger_01 import log_info, log_warning, log_debug
from scripts.recoleccionDatosAPI.searchAPI.config_00 import PAGE_SIZE, HEADERS, CONSULTAS_PATH, BASE_DIR, BASE_URL, max_pages
from scripts.recoleccionDatosAPI.searchAPI.data_io_03 import (
    guardar_anio_procesado,
    guardar_articulos,
    guardar_articulos_temporalmente,
    cargar_articulos_temporales
)
from scripts.recoleccionDatosAPI.searchAPI.scopus_request_04 import hacer_request, hacer_request_cursor, guardar_cursor, cargar_cursor, eliminar_cursor
from scripts.recoleccionDatosAPI.searchAPI.config_00 import MAX_PAGINAS_VACIAS_CONSECUTIVAS

def procesar_anio(anio, sleep_entre_llamadas):
    log_info(f"\n🗕 Procesando año {anio}")
    query_base = f"SUBJAREA(MATH) AND DOCTYPE(ar) AND PUBYEAR IS {anio}"

    params = {"query": query_base, "start": 0, "count": 1}
    data = hacer_request(params)
    total_real = int(data.get("search-results", {}).get("opensearch:totalResults", "0"))

    if total_real == 0:
        log_info(f"📂 No se encontraron artículos para el año {anio}.")
        guardar_anio_procesado(anio)
        return 0

    log_info(f"📊 Total real de artículos en Scopus para {anio}: {total_real}")

    if total_real > max_pages * PAGE_SIZE:
        log_info("🔀 Usando Cursor Pagination...")
        articulos = procesar_con_cursor(query_base, sleep_entre_llamadas, anio)
    else:
        log_info("📄 Usando paginación por start...")
        articulos = procesar_con_start(query_base, total_real, sleep_entre_llamadas)

    total_descargados = guardar_articulos(articulos, anio)
    guardar_anio_procesado(anio)
    log_info(f"✅ Año {anio} guardado con {total_descargados} artículos.")
    eliminar_cursor(anio)
    log_info(f"🧹 Cursor eliminado tras guardar exitosamente el año {anio}.")
    return total_descargados

def procesar_con_start(query_base, total_real, sleep_entre_llamadas):
    max_articulos = min(total_real, max_pages * PAGE_SIZE)
    total_paginas = (max_articulos + PAGE_SIZE - 1) // PAGE_SIZE

    articulos = []
    for page in tqdm(range(total_paginas), desc="Paginando", ncols=100):
        start = page * PAGE_SIZE
        params = {"query": query_base, "start": start, "count": PAGE_SIZE}
        data = hacer_request(params)
        resultados = data.get("search-results", {}).get("entry", [])
        mapeados = mapear_resultados(resultados)
        log_info(f"🧹 Página {page + 1}: {len(mapeados)} artículos mapeados")
        articulos += mapeados
        sleep(sleep_entre_llamadas)
    return articulos

def procesar_con_cursor(query_base, sleep_entre_llamadas, anio):
    articulos = cargar_articulos_temporales(anio)
    eids_existentes = {art["eid"] for art in articulos if "eid" in art}

    cursor = cargar_cursor(anio)
    log_debug(f"📌 Cursor inicial cargado: {cursor}")
    pagina = len(articulos) // PAGE_SIZE
    vacias_consecutivas = 0

    while True:
        try:
            guardar_cursor(anio, cursor)
            data = hacer_request_cursor(query_base, cursor)
            resultados = data.get("search-results", {}).get("entry", [])
            mapeados = mapear_resultados(resultados)
            pagina += 1

            nuevos_mapeados = [a for a in mapeados if a.get("eid") and a["eid"] not in eids_existentes]
            eids_existentes.update(a["eid"] for a in nuevos_mapeados)

            log_info(f"🗭 Cursor página {pagina}: {len(nuevos_mapeados)} artículos nuevos")

            if len(nuevos_mapeados) == 0:
                vacias_consecutivas += 1
                if vacias_consecutivas >= MAX_PAGINAS_VACIAS_CONSECUTIVAS:
                    log_warning(f"⚠️ Fin aparente de resultados: {MAX_PAGINAS_VACIAS_CONSECUTIVAS} páginas vacías seguidas. Corte preventivo.")
                    break
            else:
                vacias_consecutivas = 0

            articulos += nuevos_mapeados
            guardar_articulos_temporalmente(anio, nuevos_mapeados)
            sleep(sleep_entre_llamadas)

            cursor_next = data.get("search-results", {}).get("cursor", {}).get("@next")
            if not cursor_next:
                break

            cursor = cursor_next

        except KeyboardInterrupt:
            log_info("⏹ Interrupción manual durante descarga por cursor.")
            guardar_cursor(anio, cursor)
            raise

    return articulos

def mapear_resultados(resultados):
    return [{
        "eid": item.get("eid", ""),
        "title": item.get("dc:title", ""),
        "first_author_name": item.get("dc:creator", ""),
        "first_author_id": item.get("author", [{}])[0].get("authid", "") if item.get("author") else "",
        "all_authors": "; ".join([a.get("authname", "") for a in item.get("author", [])]) if item.get("author") else "",
        "author_ids": "; ".join([a.get("authid", "") for a in item.get("author", [])]) if item.get("author") else "",
        "cover_date": item.get("prism:coverDate", ""),
        "doi": item.get("prism:doi", ""),
        "cited_by": item.get("citedby-count", "")
    } for item in resultados]