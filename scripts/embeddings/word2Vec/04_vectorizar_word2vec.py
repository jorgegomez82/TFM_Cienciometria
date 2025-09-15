"""
04_vectorize_documents.py — Alineado con 02/03
- Carga tokens desde 02_preprocesado_{BASE}.pkl (preferente) o 02_preprocesado_filtrado_{BASE}.pkl (fallback)
- Usa modelo global (TODO) si existe en modo década; si no, el local
- Vectoriza cada documento con media de embeddings (float32)
- Calcula cobertura por documento
- Guarda PKL de salida + CSV resumen
"""

import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm
from embeddings.config_paths import TMP_DIR

def parse_args_strict():
    """
    Acepta únicamente:
      1) --todo
      2) --decada YYYY_YYYY
    Devuelve: {"todo": bool, "decada": str}
    """
    argv = sys.argv[1:]
    if argv == ["--todo"]:
        return {"todo": True, "decada": "TODO"}
    if len(argv) == 2 and argv[0] == "--decada" and re.fullmatch(r"\d{4}_\d{4}", argv[1]):
        return {"todo": False, "decada": argv[1]}
    raise SystemExit(
        "Uso permitido:\n"
        "  py 04_vectorize_documents.py --todo\n"
        "  py 04_vectorize_documents.py --decada 1960_1969\n"
    )

args = parse_args_strict()
BASE = args["decada"]

# ---------- Rutas dinámicas ----------
def _tokens_path(base: str):
    # Coherente con 03: preferimos 02_preprocesado_{BASE}.pkl
    p02  = os.path.join(TMP_DIR, f"02_preprocesado_{base}.pkl")
    p02f = os.path.join(TMP_DIR, f"02_preprocesado_filtrado_{base}.pkl")
    if os.path.exists(p02):
        return p02
    if os.path.exists(p02f):
        return p02f
    raise FileNotFoundError(f"❌ No se encontró {p02} ni {p02f}. Ejecuta 02 primero.")

def _model_path(base: str):
    global_model = os.path.join(TMP_DIR, "03_word2vec_TODO.model")
    # En modo década, si existe el global lo usamos para comparabilidad
    if base != "TODO" and os.path.exists(global_model):
        return global_model
    local_model = os.path.join(TMP_DIR, f"03_word2vec_{base}.model")
    if not os.path.exists(local_model):
        msg = f"❌ No existe el modelo {local_model}"
        if base != "TODO":
            msg += f"\n   (Tampoco se usó el global porque no existe: {global_model})"
        raise FileNotFoundError(msg)
    return local_model

OUTPUT_PATH = os.path.join(TMP_DIR, f"04_vectores_word2vec_{BASE}.pkl")
RESUMEN_CSV = os.path.join(TMP_DIR, f"04_vectores_resumen_word2vec_{BASE}.csv")

# Si un doc no tiene ningún token en el vocab, rellena con vector cero
FILL_EMPTY = True

# ---------- Funciones ----------
def coverage_ratio(tokens_list, wv_keyset) -> float:
    if not tokens_list:
        return 0.0
    hits = sum(1 for w in tokens_list if w in wv_keyset)
    return hits / len(tokens_list)

def vectorize(tokens_list, wv, wv_keyset, dim: int):
    if not tokens_list:
        return (0.0, np.zeros(dim, dtype=np.float32) if FILL_EMPTY else None)
    # filtrar solo tokens presentes en el vocab
    present = [w for w in tokens_list if w in wv_keyset]
    cov = (len(present) / len(tokens_list)) if tokens_list else 0.0
    if present:
        # get_vector es algo más rápido que wv[w] en algunos builds
        vecs = [wv.get_vector(w) for w in present]
        # media en float32
        return (cov, np.asarray(vecs, dtype=np.float32).mean(axis=0))
    return (cov, np.zeros(dim, dtype=np.float32) if FILL_EMPTY else None)

# ---------- Principal ----------
def main():
    print("📦 Resolviendo rutas de entrada...")
    TOKENS_PATH = _tokens_path(BASE)
    MODEL_PATH  = _model_path(BASE)
    print(f"   Tokens: {TOKENS_PATH}")
    print(f"   Modelo: {MODEL_PATH}")

    print("📥 Cargando tokens preprocesados...")
    with open(TOKENS_PATH, "rb") as f:
        df = pickle.load(f)

    if "tokens" not in df.columns or df["tokens"].isnull().all():
        raise ValueError("❌ La columna 'tokens' está vacía o no existe.")

    print("🧠 Cargando modelo Word2Vec...")
    model = Word2Vec.load(MODEL_PATH)
    wv = model.wv
    dim = wv.vector_size
    # set/dict para membership O(1)
    wv_keyset = set(wv.key_to_index.keys())
    print(f"   Dimensión de vector en el modelo: {dim}")
    print(f"   Vocabulario del modelo: {len(wv_keyset):,}".replace(",", " "))

    # 🔍 Chequeo contra PKL previo (dimensión consistente)
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "rb") as f:
            prev_df = pickle.load(f)
        if "vector" in prev_df.columns and prev_df["vector"].notnull().any():
            first_vec = next((v for v in prev_df["vector"] if v is not None), None)
            if first_vec is not None and len(first_vec) != dim:
                raise ValueError(
                    f"❌ Dimensión de vector inconsistente.\n"
                    f"   PKL previo: {len(first_vec)} dimensiones\n"
                    f"   Modelo actual: {dim} dimensiones\n"
                    f"   Sugerencia: Regenera el PKL ejecutando este script desde cero."
                )

    # 🔄 Cobertura + vectorización en una pasada (más rápido que progress_apply)
    coverages = []
    vectors = []
    it = df["tokens"].tolist()
    print("🔄 Calculando cobertura y vectores por documento...")
    for toks in tqdm(it, total=len(it), unit="doc"):
        cov, vec = vectorize(toks, wv, wv_keyset, dim)
        coverages.append(cov)
        vectors.append(vec)

    df["coverage_ratio"] = coverages
    df["vector"] = vectors

    # Informe de cobertura
    cov_mean = float(np.nanmean([c for c in coverages if c is not None])) if coverages else 0.0
    print(f"   Cobertura promedio: {cov_mean:.3f}")

    if FILL_EMPTY:
        zeros = int(sum(1 for v in vectors if isinstance(v, np.ndarray) and np.allclose(v, 0)))
        print(f"ℹ️ Docs con vector nulo: {zeros} de {len(df)} ({zeros/len(df):.2%})")
    else:
        before = len(df)
        df = df[df["vector"].notnull()]
        print(f"⚠️ Documentos descartados por vector None: {before - len(df)}")

    print("📆 Asegurando tipo datetime en cover_date (si existe)...")
    if "cover_date" in df.columns:
        df["cover_date"] = pd.to_datetime(df["cover_date"], errors="coerce")

    print("↕️ Ordenando (si existen columnas)...")
    sort_cols = [c for c in ["cover_date", "eid"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    print("💾 Guardando PKL con vectores...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(df, f)
    print(f"📁 Guardado en: {OUTPUT_PATH}")

    print("📑 Guardando CSV resumido...")
    cols = [c for c in ["eid", "title", "cover_date", "coverage_ratio"] if c in df.columns]
    if cols:
        df[cols].to_csv(RESUMEN_CSV, index=False)
        print(f"📁 Guardado en: {RESUMEN_CSV}")
    else:
        print("ℹ️ Columnas para resumen no disponibles; se omite CSV.")

    print(f"✅ Total documentos vectorizados: {len(df)}")

if __name__ == "__main__":
    main()
