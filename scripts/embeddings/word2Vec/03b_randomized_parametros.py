# 03b_randomized_parametros.py
# -----------------------------------------------------------
# Búsqueda aleatoria de hiperparámetros para Word2Vec
# - CLI estricta: --todo  |  --decada YYYY_YYYY
# - Carga tokens desde 02b (preferente) o 02 (fallback)
# - Lee combinaciones desde parametros_randomized_word2vec.csv
# - Entrena cada combinación, calcula métricas y guarda modelo
# - Resultados incrementales + final ordenado por "score"
# - Métricas extra: cobertura de vocabulario sobre muestra, tokens/seg, tamaño .kv
# - Reanudable: si el modelo existe, lo salta (a menos que --overwrite)
# -----------------------------------------------------------

import os
import re
import sys
import time
import json
import pickle
import math
import random
from collections import Counter

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from embeddings.config_paths import TMP_DIR

# ============== CLI estricta (igual estilo que 03_train_word2vec.py) ==============
USING_COMMON = True
try:
    from tfm_common import parse_args_strict  # type: ignore
except Exception:
    USING_COMMON = False

    def parse_args_strict():
        """
        Acepta únicamente:
          1) --todo
          2) --decada YYYY_YYYY
        Opcional:
          --overwrite  (fuerza reentrenar aunque exista archivo)
          --sample N   (muestra de documentos para métrica de cobertura; default 5000)
        Devuelve: dict con claves: todo (bool), decada (str|None), overwrite (bool), sample (int)
        """
        argv = sys.argv[1:]
        args = {"todo": False, "decada": None, "overwrite": False, "sample": 5000}

        # flags opcionales
        if "--overwrite" in argv:
            args["overwrite"] = True
            argv = [a for a in argv if a != "--overwrite"]

        if "--sample" in argv:
            i = argv.index("--sample")
            try:
                args["sample"] = int(argv[i + 1])
            except Exception:
                raise SystemExit("❌ Uso: --sample N (entero positivo)")
            argv = argv[:i] + argv[i + 2:]

        # principales
        if argv == ["--todo"]:
            args["todo"] = True
            args["decada"] = None
            return args

        if len(argv) == 2 and argv[0] == "--decada" and re.fullmatch(r"\d{4}_\d{4}", argv[1]):
            args["todo"] = False
            args["decada"] = argv[1]
            return args

        raise SystemExit(
            "Uso permitido (solo 2 formas):\n"
            "  1) py 03b_randomized_parametros.py --todo [--overwrite] [--sample N]\n"
            "  2) py 03b_randomized_parametros.py --decada 1960_1969 [--overwrite] [--sample N]\n"
        )

# ============== Utilidades de carga/paths ==============
def _preprocess_path(base: str):
    """Retorna ruta preferente 02b y fallback 02."""
    p02b = os.path.join(TMP_DIR, f"02_preprocesado_filtrado_{base}.pkl")
    p02  = os.path.join(TMP_DIR, f"02_preprocesado_{base}.pkl")
    path = p02b if os.path.exists(p02b) else p02
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No se encontró {p02b} ni {p02}. Ejecuta 02/02b primero.")
    return path

def load_corpus(base: str):
    """Carga DF preprocesado (02b o 02) y devuelve lista de tokens por doc."""
    path = _preprocess_path(base)
    print(f"📦 Cargando tokens desde: {path}")
    with open(path, "rb") as f:
        df = pickle.load(f)
    if "tokens" not in df.columns or df["tokens"].isnull().all():
        raise ValueError("❌ La entrada no contiene una columna válida 'tokens'.")
    corpus = df["tokens"].tolist()
    print(f"✅ Corpus cargado con {len(corpus)} documentos.")
    return corpus

def base_tag(args):
    return "TODO" if args["todo"] else args["decada"]

# ============== Métricas auxiliares ==============
def analyze_vocab_quick(corpus, thresholds=(1, 2, 3, 5)):
    flat = (t for doc in corpus for t in doc)
    counts = Counter(flat)
    total = len(counts)
    print("\n🔍 Análisis rápido de vocabulario (corpus de entrada):")
    print(f"   Palabras únicas totales (sin filtrar): {total}")
    for th in thresholds:
        kept = sum(1 for _, c in counts.items() if c >= th)
        print(f"   - min_count>={th}: {kept} (descartadas: {total - kept})")

def sample_docs(corpus, k=5000, seed=42):
    if k <= 0:
        return []
    rnd = random.Random(seed)
    if len(corpus) <= k:
        return corpus
    idx = rnd.sample(range(len(corpus)), k)
    return [corpus[i] for i in idx]

def coverage_on_sample(model, docs_sample):
    """Cobertura promedio: % de tokens presentes en el vocab del modelo por documento."""
    if not docs_sample:
        return np.nan
    in_vocab_ratios = []
    wv = model.wv
    for toks in docs_sample:
        if not toks:
            continue
        hit = sum(1 for t in toks if t in wv)
        in_vocab_ratios.append(hit / max(1, len(toks)))
    return float(np.mean(in_vocab_ratios)) if in_vocab_ratios else np.nan

def kv_disk_megabytes(model):
    """Aproximación del tamaño del .kv (solo embeddings)."""
    vocab = len(model.wv)
    dim = model.vector_size
    bytes_ = vocab * dim * 4  # float32
    return round(bytes_ / (1024 ** 2), 2)

# ============== Principal ==============
def main():
    args = parse_args_strict()
    base = base_tag(args)

    # Archivos y directorios destino
    MODELS_DIR = os.path.join(TMP_DIR, f"random_models_{base}")
    RESULTS_PATH = os.path.join(TMP_DIR, f"random_search_resultados_{base}.csv")
    PARAMS_CSV = os.path.join(TMP_DIR, "parametros_randomized_word2vec.csv")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Cargar corpus
    corpus = load_corpus(base)
    analyze_vocab_quick(corpus)

    # Cargar combinaciones
    if not os.path.exists(PARAMS_CSV):
        raise FileNotFoundError(f"❌ No existe {PARAMS_CSV}. Genera ese CSV con las combinaciones a evaluar.")
    params_df = pd.read_csv(PARAMS_CSV)
    if params_df.empty:
        raise ValueError("❌ El CSV de parámetros está vacío.")
    # Coerción de tipos estándar esperados
    for col in ["vector_size", "window", "min_count", "sg", "epochs"]:
        if col not in params_df.columns:
            raise ValueError(f"❌ Falta la columna '{col}' en {PARAMS_CSV}.")
        params_df[col] = params_df[col].astype(int)

    # Muestra para métrica de cobertura
    docs_sample = sample_docs(corpus, k=args["sample"], seed=42)
    print(f"🧪 Muestra para cobertura: {len(docs_sample)} documentos.")

    # Si existen resultados previos, los cargamos para reanudar
    done_ids = set()
    resultados_prev = None
    if os.path.exists(RESULTS_PATH):
        try:
            resultados_prev = pd.read_csv(RESULTS_PATH)
            done_ids = set(resultados_prev["modelo_id"].astype(int).tolist())
            print(f"🔁 Reanudando: {len(done_ids)} combinaciones ya procesadas.")
        except Exception:
            print("⚠️ No se pudo leer el CSV previo de resultados; se creará uno nuevo.")
            resultados_prev = None
            done_ids = set()

    resultados = [] if resultados_prev is None else resultados_prev.to_dict("records")

    # Búsqueda
    total = len(params_df)
    for idx, row in params_df.iterrows():
        modelo_id = int(idx + 1)
        vsize   = int(row["vector_size"])
        window  = int(row["window"])
        mcount  = int(row["min_count"])
        sg      = int(row["sg"])
        epochs  = int(row["epochs"])
        seed    = int(row["seed"]) if "seed" in row and not pd.isna(row["seed"]) else 42
        workers = max(1, (os.cpu_count() or 2) - 1)

        # Nombre de archivo informativo
        tag = f"vs{vsize}_w{window}_mc{mcount}_sg{sg}_e{epochs}"
        model_path = os.path.join(MODELS_DIR, f"modelo_{modelo_id}_{tag}.model")

        if (not args["overwrite"]) and os.path.exists(model_path):
            if modelo_id in done_ids:
                print(f"⏭️  [{modelo_id}/{total}] Ya existe y en resultados: {os.path.basename(model_path)}. Saltando.")
                continue
            else:
                print(f"⏭️  [{modelo_id}/{total}] Ya existe el archivo {os.path.basename(model_path)}. Saltando (usa --overwrite para forzar).")
                continue

        print(f"\n🔧 [{modelo_id}/{total}] Entrenando: vector_size={vsize}, window={window}, min_count={mcount}, sg={sg}, epochs={epochs}, seed={seed}")

        # Entrenamiento
        t0 = time.time()
        try:
            model = Word2Vec(
                sentences=corpus,
                vector_size=vsize,
                window=window,
                min_count=mcount,
                sg=sg,
                epochs=epochs,
                workers=workers,
                seed=seed
            )
        except Exception as e:
            print(f"❌ Error entrenando combinación #{modelo_id}: {e}")
            resultados.append({
                "modelo_id": modelo_id,
                "vector_size": vsize,
                "window": window,
                "min_count": mcount,
                "sg": sg,
                "epochs": epochs,
                "workers": workers,
                "seed": seed,
                "status": "error",
                "error_msg": str(e)
            })
            # Guardado incremental
            pd.DataFrame(resultados).to_csv(RESULTS_PATH, index=False)
            continue

        elapsed = time.time() - t0

        # Métricas
        vocab_size = len(model.wv)
        norms = [np.linalg.norm(model.wv[w]) for w in model.wv.index_to_key]
        avg_norm = float(np.mean(norms)) if norms else 0.0

        # Aproximación de throughput
        # tokens procesados ~ (suma de longitudes por doc) * epochs
        # calculamos con una estimación rápida (promedio tokens/doc)
        avg_len = np.mean([len(d) for d in docs_sample]) if docs_sample else np.nan
        tokens_totales_est = (avg_len * len(corpus) * epochs) if not math.isnan(avg_len) else np.nan
        tokens_por_seg = (tokens_totales_est / elapsed) if (tokens_totales_est and elapsed > 0) else np.nan

        # Cobertura de vocabulario en muestra
        cov_sample = coverage_on_sample(model, docs_sample)

        # Tamaño estimado del .kv
        kv_mb = kv_disk_megabytes(model)

        # Score (mantenemos tu idea base pero la estabilizamos)
        # Combina vocab_size (cobertura), avg_norm (escala estable) y cobertura de muestra
        # Pesos heurísticos para priorizar cobertura efectiva:
        score = (vocab_size * avg_norm) * (1.0 + (cov_sample if not math.isnan(cov_sample) else 0.0))

        # Guardar modelo individual
        try:
            model.save(model_path)
            print(f"✅ Modelo guardado en: {model_path}")
        except Exception as e:
            print(f"⚠️ No se pudo guardar el modelo ({e}). Se registrará como 'trained_not_saved'.")

        # Registro de resultados
        resultados.append({
            "modelo_id": modelo_id,
            "vector_size": vsize,
            "window": window,
            "min_count": mcount,
            "sg": sg,
            "epochs": epochs,
            "workers": workers,
            "seed": seed,
            "vocab_size": vocab_size,
            "avg_vector_norm": round(avg_norm, 6),
            "coverage_sample": round(cov_sample, 6) if cov_sample == cov_sample else np.nan,  # NaN safe
            "kv_size_mb_est": kv_mb,
            "tiempo_seg": round(elapsed, 2),
            "tokens_por_seg_est": round(tokens_por_seg, 2) if tokens_por_seg == tokens_por_seg else np.nan,
            "score": round(score, 6),
            "status": "ok" if os.path.exists(model_path) else "trained_not_saved",
            "model_path": model_path if os.path.exists(model_path) else ""
        })

        # Guardado incremental para resiliencia
        pd.DataFrame(resultados).to_csv(RESULTS_PATH, index=False)

    # Orden final por score desc
    resultados_df = pd.DataFrame(resultados)
    if not resultados_df.empty and "score" in resultados_df.columns:
        resultados_df = resultados_df.sort_values(by="score", ascending=False)
        resultados_df.to_csv(RESULTS_PATH, index=False)

    print("\n🏁 Búsqueda completada.")
    print(f"📝 Resultados guardados en: {RESULTS_PATH}")
    print(f"📂 Modelos en: {MODELS_DIR}")

if __name__ == "__main__":
    main()
