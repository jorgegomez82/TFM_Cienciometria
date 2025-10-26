--------------------------------------------------------------------------------
Conjunto de Datos: Lista de EIDs de Scopus para TFM en Cienciometría
--------------------------------------------------------------------------------

Autor: Jorge Gómez Gómez
Fecha de Publicación en Zenodo: 25-10-2025
DOI: https://doi.org/10.5281/zenodo.17445712

## Descripción

Este conjunto de datos contiene los identificadores únicos (EIDs) de artículos científicos extraídos de la base de datos Scopus. La lista de EIDs se proporciona en formato CSV y es la base para la investigación realizada en el Trabajo de Fin de Máster Cienciometría: Índice de novedad en publicaciones científicas del campo de las matemáticas para la Universidad Internacional de Valencia.

El propósito de publicar estos EIDs es garantizar la total reproducibilidad del corpus de investigación, permitiendo a otros investigadores reconstruir el conjunto de datos original respetando los términos de uso de la API de Scopus.

## Metodología de Recopilación

Los EIDs fueron recopilados mediante la API de Búsqueda de Scopus (Search API) entre julio y agosto de 2025. El proceso se detalla en los scripts disponibles en el siguiente repositorio de GitHub.

* **Repositorio de Código:** https://github.com/jorgegomez82/TFM_Cienciometria

Los criterios de búsqueda aplicados, extraídos del script `procesador_05.py`, fueron los siguientes:

* **Consulta Principal:** `SUBJAREA(MATH) AND DOCTYPE(ar) AND PUBYEAR IS {anio}`
    * **Áreas Temáticas:** Matemáticas (MATH).
    * **Tipo de Documento:** Artículo de revista (ar).

* **Rango Temporal:** La búsqueda se realizó desde 1826 hasta 2009

## Contenido del Archivo

* **`articulos_scopus_eid.csv`**: Un archivo de texto en formato CSV con una única columna (`EID`). Cada fila contiene el identificador de un artículo que cumple con los criterios de búsqueda.

## Uso y Reproducibilidad

Para reconstruir el conjunto de datos completo (incluyendo títulos, resúmenes, etc.), se debe utilizar la lista de EIDs de este dataset y consultar la API de Recuperación de Resúmenes de Scopus (Abstract Retrieval API), tal como se implementa en el script `enriquecer_desde_eid.py` del repositorio mencionado.

## Cómo Citar

Gomez Gomez, J. C. (2025). eid_articulos_matematicos. Zenodo. https://doi.org/10.5281/zenodo.17445712