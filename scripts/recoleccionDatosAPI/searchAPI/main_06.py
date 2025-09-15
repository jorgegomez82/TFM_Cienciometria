# main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from time import time
from scripts.recoleccionDatosAPI.searchAPI.config_00 import START_YEAR, END_YEAR, sleep_entre_llamadas
from scripts.recoleccionDatosAPI.searchAPI.logger_01 import log_info
from scripts.recoleccionDatosAPI.searchAPI.utils_02 import formato_segundos, esperar_segundos
from scripts.recoleccionDatosAPI.searchAPI.data_io_03 import (
    cargar_anios_procesados,
    guardar_contador_consultas,
    registrar_progreso
)
from scripts.recoleccionDatosAPI.searchAPI.procesador_05 import procesar_anio

if __name__ == "__main__":
    try:
        inicio_descarga = time()
        anios_procesados = cargar_anios_procesados()
        decadas_completadas = 0
        decadas_totales = (END_YEAR // 10) - (START_YEAR // 10) + 1
        total_articulos = 0

        anio = START_YEAR
        
        log_info("⏳ Iniciando reanudación. Esperando 5 minutos para evitar bloqueo...")
        esperar_segundos(300)

        while anio <= END_YEAR:
            if anio in anios_procesados:
                log_info(f"⏩ Año {anio} ya procesado. Saltando.")
                anio += 1
                continue

            try:
                total_articulos += procesar_anio(anio, sleep_entre_llamadas)

                if anio % 10 == 9:
                    decada_inicio = anio - 9
                    decada_fin = anio
                    decadas_completadas += 1
                    duracion = time() - inicio_descarga
                    tiempo_promedio = duracion / decadas_completadas
                    restantes = decadas_totales - decadas_completadas
                    estimado_restante = tiempo_promedio * restantes

                    registrar_progreso(
                        decada_inicio,
                        decada_fin,
                        duracion,
                        descargados=total_articulos,
                        consultas=0,
                        tiempo_estimado=estimado_restante,
                        restantes=restantes
                    )

                    # esperar_segundos(900)  # 🔄 Eliminado para máxima velocidad

                anio += 1

            except KeyboardInterrupt:
                try:
                    respuesta = input("\n🛑 ¿Deseas detener completamente el proceso? (s/n): ").strip().lower()
                    if respuesta == 's':
                        log_info("👋 Proceso finalizado por el usuario.")
                        break
                    else:
                        log_info("↪️ Continuando con el mismo año tras interrupción...")
                except Exception:
                    log_info("⚠️ Error durante confirmación de interrupción. Continuando por defecto.")

        duracion_final = time() - inicio_descarga
        log_info("\n✅ Descarga completada.")
        log_info(f"⏱ Duración total: {formato_segundos(duracion_final)}")
        log_info(f"📚 Artículos totales descargados: {total_articulos}")
        guardar_contador_consultas(0)
        log_info("=" * 60)

    except Exception as e:
        log_info(f"❌ Error crítico: {type(e).__name__} - {e}")
        import traceback
        log_info(traceback.format_exc())