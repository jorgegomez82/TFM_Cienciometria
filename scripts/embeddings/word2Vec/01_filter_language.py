"""
01_filter_language.py (solo 2 formas de ejecución: --todo o --decada YYYY_YYYY)
Filtra artículos cuyo abstract y título están en inglés usando fastText (lid.176.ftz).
Ignora columnas: subject_areas, num_subdisciplinas, idioma (no se escriben en salidas).
"""
import os
import sys
import re
import glob
import urllib.request
from collections import Counter

import pandas as pd
import fasttext
from tqdm import tqdm

# Rutas del proyecto
from embeddings.config_paths import ENRIQUECIDO_DIR, TMP_DIR, IDIOMA_DIR

MODEL_NAME = "lid.176.ftz"
MODEL_PATH = os.path.join(IDIOMA_DIR, MODEL_NAME)
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

UMBRAL_ABSTRACT_EN = 0.60
MIN_WORDS_ABSTRACT = 3

# Tamaños de bloque fijos (filas) para leer CSV
CHUNK_TODO   = 400_000
CHUNK_DECADA = 250_000

# Columnas que NO queremos en las salidas
IGNORED_COLS = {"subject_areas", "num_subdisciplinas", "idioma"}

def asegurar_modelo():
    if not os.path.exists(MODEL_PATH):
        print("📥 Modelo fastText no encontrado. Descargando...")
        os.makedirs(IDIOMA_DIR, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Modelo descargado.")
    print("📦 Cargando modelo fastText...")
    return fasttext.load_model(MODEL_PATH)

def predict_list(model, textos):
    """
    Predice en lote: devuelve dos listas (langs, confs) del mismo largo que textos.
    """
    labels, probs = model.predict(textos)
    langs = [lbls[0].replace("__label__", "") if lbls else "" for lbls in labels]
    confs = [float(ps[0]) if ps else 0.0 for ps in probs]
    return langs, confs

def actualizar_counter(counter: Counter, serie: pd.Series):
    vc = serie.value_counts(dropna=False)
    for k, v in vc.items():
        counter[k] += int(v)

def preparar_salida(path):
    # Elimina archivo previo para escribir header una vez
    if os.path.exists(path):
        os.remove(path)

def write_chunk_df(path: str, df: pd.DataFrame, header_cols: list):
    """
    Escribe un chunk DataFrame:
      - Elimina IGNORED_COLS si existen.
      - Reordena/expande columnas según header_cols (faltantes -> "").
      - Escribe en modo append con header solo la primera vez.
    """
    if df is None or df.empty:
        return
    # quitar columnas ignoradas
    df2 = df.drop(columns=list(IGNORED_COLS), errors="ignore").copy()

    # asegurar columnas del header maestro
    for col in header_cols:
        if col not in df2.columns:
            df2[col] = ""
    df2 = df2[header_cols]

    first_time = not os.path.exists(path)
    df2.to_csv(path, index=False, mode="a", header=first_time, encoding="utf-8")

def procesar_chunk(df_chunk: pd.DataFrame, model):
    if "abstract" not in df_chunk.columns or "title" not in df_chunk.columns:
        return pd.DataFrame(), pd.DataFrame(), Counter(), Counter()

    df = df_chunk.copy()
    # Evitar "nan" como texto: primero fillna y luego astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str).str.strip()
    df["title"]    = df["title"].fillna("").astype(str).str.strip()

    # abstracts demasiado cortos: solo por número de palabras
    s = df["abstract"].astype(str)
    mask_valid_abs = s.str.split().str.len() > MIN_WORDS_ABSTRACT

    # Predicción batch para abstracts válidos
    langs_abs = pd.Series([""] * len(df), index=df.index, dtype="object")
    confs_abs = pd.Series([0.0] * len(df), index=df.index, dtype="float32")
    if mask_valid_abs.any():
        la, ca = predict_list(model, df.loc[mask_valid_abs, "abstract"].tolist())
        langs_abs.loc[mask_valid_abs] = la
        confs_abs.loc[mask_valid_abs] = ca

    # Short-circuit: predecir título solo si el abstract ya pasó
    mask_need_title = (langs_abs == "en") & (confs_abs >= UMBRAL_ABSTRACT_EN)

    langs_title = pd.Series([""] * len(df), index=df.index, dtype="object")
    confs_title = pd.Series([0.0] * len(df), index=df.index, dtype="float32")
    if mask_need_title.any():
        lt, ct = predict_list(model, df.loc[mask_need_title, "title"].str.lower().tolist())
        langs_title.loc[mask_need_title] = lt
        confs_title.loc[mask_need_title] = ct

    # adjunta columnas
    df["lang_abstract"] = langs_abs
    df["conf_abstract"] = confs_abs
    df["lang_title"]    = langs_title
    df["conf_title"]    = confs_title

    # criterio de aceptación
    mask_ok = (df["lang_abstract"].eq("en")) & (df["conf_abstract"] >= UMBRAL_ABSTRACT_EN) & (df["lang_title"].eq("en"))
    df_ok  = df.loc[mask_ok].copy()
    df_bad = df.loc[~mask_ok].copy()

    # text final (solo para los OK)
    if not df_ok.empty:
        df_ok["text"] = (df_ok["title"].astype(str) + ". " + df_ok["abstract"].astype(str)).str.strip()

    # contadores de idiomas
    cabs = Counter(); ctit = Counter()
    actualizar_counter(cabs, df["lang_abstract"])
    actualizar_counter(ctit, df["lang_title"])

    return df_ok, df_bad, cabs, ctit

def guardar_resumen_idiomas(cabs, ctit, path_csv, total_registros):
    sin_abs  = cabs.get("", 0)
    sin_tit  = ctit.get("", 0)
    idiomas  = sorted({k for k in set(cabs) | set(ctit) if k})

    rows = [{"idioma": lang,
             "abstract_count": cabs.get(lang, 0),
             "titulo_count":   ctit.get(lang, 0)} for lang in idiomas]

    if sin_abs or sin_tit:
        rows.append({"idioma": "(sin_deteccion)",
                     "abstract_count": sin_abs,
                     "titulo_count":   sin_tit})

    rows.append({"idioma": "TOTAL",
                 "abstract_count": total_registros,
                 "titulo_count":   total_registros})
    pd.DataFrame(rows)[["idioma","abstract_count","titulo_count"]].to_csv(path_csv, index=False, encoding="utf-8")

# -------------------------------
# Header maestro (modo TODO)
# -------------------------------

def construir_header_maestro(files, incluir_text: bool = True):
    """
    Devuelve (header_filtrado, header_descartado) unificando columnas base y
    agregando las generadas por el script. Excluye IGNORED_COLS.
    """
    base_cols = set()
    for f in files:
        cols = pd.read_csv(f, nrows=0).columns
        base_cols.update(cols)

    # columnas añadidas por el script
    if incluir_text:
        extras_filtrado = ["text", "lang_abstract", "conf_abstract", "lang_title", "conf_title"]
    else:
        extras_filtrado = ["lang_abstract", "conf_abstract", "lang_title", "conf_title"]
    extras_descartado = ["lang_abstract", "conf_abstract", "lang_title", "conf_title"]

    # orden opcional (priorizar campos clave si existen)
    prioridad = [c for c in ["eid", "cover_date", "doi", "citedby_count", "keywords", "title", "abstract"] if c in base_cols]
    resto     = sorted([c for c in base_cols if c not in prioridad])

    header_filtrado = [c for c in (prioridad + resto + extras_filtrado) if c not in IGNORED_COLS]
    header_desc     = [c for c in (prioridad + resto + extras_descartado) if c not in IGNORED_COLS]
    return header_filtrado, header_desc

# -------------------------------
# Flujo principal
# -------------------------------

def run_modo_decada(decada: str, model):
    in_csv = os.path.join(ENRIQUECIDO_DIR, f"enriquecido_articulos_scopus_{decada}.csv")
    out_filtrado     = os.path.join(TMP_DIR, f"01_filtrado_abstract_ingles_{decada}.csv")
    out_descartados  = os.path.join(TMP_DIR, f"01_articulos_descartados_no_ingles_{decada}.csv")
    out_resumen      = os.path.join(TMP_DIR, f"01_resumen_idiomas_detectados_{decada}.csv")

    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"❌ No se encontró el archivo: {in_csv}")

    preparar_salida(out_filtrado)
    preparar_salida(out_descartados)

    # Header para esta década (solo con columnas de este archivo)
    header_filtrado, header_desc = construir_header_maestro([in_csv], incluir_text=True)

    cabs_total, ctit_total = Counter(), Counter()
    total_registros = 0

    for chunk in tqdm(pd.read_csv(in_csv, chunksize=CHUNK_DECADA, engine="c", encoding="utf-8",
                on_bad_lines="skip", low_memory=False), desc=f"Década {decada}", ascii=True):
        df_ok, df_bad, cabs, ctit = procesar_chunk(chunk, model)
        total_registros += len(chunk)
        cabs_total.update(cabs); ctit_total.update(ctit)
        write_chunk_df(out_filtrado,    df_ok,  header_filtrado)
        write_chunk_df(out_descartados, df_bad, header_desc)

    guardar_resumen_idiomas(cabs_total, ctit_total, out_resumen, total_registros)
    print(f"✅ Guardado: {out_filtrado}")
    print(f"✅ Guardado: {out_descartados}")
    print(f"✅ Guardado: {out_resumen}")
    
    if not os.path.exists(out_filtrado):    pd.DataFrame(columns=header_filtrado).to_csv(out_filtrado, index=False)
    if not os.path.exists(out_descartados): pd.DataFrame(columns=header_desc).to_csv(out_descartados, index=False)

def run_modo_todo(model):
    patrones = os.path.join(ENRIQUECIDO_DIR, "enriquecido_articulos_scopus_*.csv")
    files = sorted(glob.glob(patrones))
    if not files:
        raise FileNotFoundError(f"❌ No se encontraron archivos con patrón: {patrones}")

    out_filtrado_todo    = os.path.join(TMP_DIR, "01_filtrado_abstract_ingles_TODO.csv")
    out_descartados_todo = os.path.join(TMP_DIR, "01_articulos_descartados_no_ingles_TODO.csv")
    out_resumen_todo     = os.path.join(TMP_DIR, "01_resumen_idiomas_detectados_TODO.csv")

    preparar_salida(out_filtrado_todo)
    preparar_salida(out_descartados_todo)

    # Header maestro unificado (excluye columnas ignoradas)
    header_filtrado, header_desc = construir_header_maestro(files, incluir_text=True)

    cabs_total, ctit_total = Counter(), Counter()
    total_registros = 0

    for in_csv in tqdm(files, desc="Archivos", ascii=True, position=0):
        for chunk in tqdm(
            pd.read_csv(in_csv, chunksize=CHUNK_TODO, engine="c", encoding="utf-8",
                        on_bad_lines="skip", low_memory=False),
            desc=os.path.basename(in_csv),
            ascii=True,
            position=1,         # barra “interna”
            leave=False,        # evitar que queden múltiples barras impresas
            mininterval=0.1
        ):
            df_ok, df_bad, cabs, ctit = procesar_chunk(chunk, model)
            total_registros += len(chunk)
            cabs_total.update(cabs); ctit_total.update(ctit)
            write_chunk_df(out_filtrado_todo,    df_ok,  header_filtrado)
            write_chunk_df(out_descartados_todo, df_bad, header_desc)

    guardar_resumen_idiomas(cabs_total, ctit_total, out_resumen_todo, total_registros)
    print(f"\n✅ Maestro guardado: {out_filtrado_todo}")
    print(f"✅ Maestro guardado: {out_descartados_todo}")
    print(f"✅ Resumen maestro: {out_resumen_todo}")

    if not os.path.exists(out_filtrado_todo):    pd.DataFrame(columns=header_filtrado).to_csv(out_filtrado_todo, index=False)
    if not os.path.exists(out_descartados_todo): pd.DataFrame(columns=header_desc).to_csv(out_descartados_todo, index=False)

def parse_args_strict():    
    argv = sys.argv[1:]    
    
    if argv == ["--todo"]:
        return {"todo": True, "decada": None}

    if len(argv) == 2 and argv[0] == "--decada":
        dec = argv[1]
        if not re.fullmatch(r"\d{4}_\d{4}", dec):
            raise SystemExit("❌ Formato inválido: --decada debe ser YYYY_YYYY (ej. 1960_1969).")
        return {"todo": False, "decada": dec}

    # Si no coincide con ninguna, mensaje de uso estricto
    msg = (
        "Uso permitido (solo 2 formas):\n"
        "  1) py 01_filter_language.py --todo\n"
        "  2) py 01_filter_language.py --decada 1960_1969\n"
    )
    raise SystemExit("❌ Argumentos no válidos.\n" + msg)

def main():
    args = parse_args_strict()
    model = asegurar_modelo()

    if args["todo"]:
        print(f"🚀 Ejecutando en modo TODO (todas las décadas). CHUNK = {CHUNK_TODO}")
        run_modo_todo(model)
    else:
        dec = args["decada"]
        print(f"🔎 Ejecutando en modo DÉCADA: {dec}. CHUNK = {CHUNK_DECADA}")
        run_modo_decada(dec, model)

if __name__ == "__main__":
    main()
