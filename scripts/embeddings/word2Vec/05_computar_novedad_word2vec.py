"""
05_compute_novelty.py (optimizado y alineado con 04)

Calcula índices de novedad por percentiles (0, 5, 10, 20) comparando cada
artículo con previos según filtros temporales opcionales:
  - --ventana_k K : solo K previos más recientes
  - --meses M     : solo previos dentro de los últimos M meses
  - --estricto    : EXCLUYE el mes actual (solo meses estrictamente anteriores)
  - --pool_min N  : tamaño mínimo del conjunto de comparación; si 'estricto' no llega, añade coetáneos del mismo mes hasta alcanzar N
  - --min_coverage C : (nuevo) exige cobertura mínima C (0..1) para usar un doc (propio y previos)

Coherente con 02/03/04/06 y CLI estricta:
  py 05_compute_novelty.py --todo [--ventana_k 50000] [--meses 24] [--estricto] [--pool_min 5000] [--min_coverage 0.05]
  py 05_compute_novelty.py --decada 1960_1969 [--ventana_k 30000] [--estricto] [--pool_min 5000] [--min_coverage 0.05]
"""

import os
import sys
import re
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from embeddings.config_paths import TMP_DIR

# -------------------------------
# CLI coherente con 03/04
# -------------------------------
def parse_args_strict():
    argv = sys.argv[1:]
    args = {
        "todo": False,
        "decada": None,
        "ventana_k": None,   # int|None
        "meses": None,       # int|None
        "estricto": False,   # bool
        "pool_min": None,    # int|None
        "min_coverage": None # float|None (0..1)
    }

    def pop_int(flag, key):
        nonlocal argv
        if flag in argv:
            i = argv.index(flag)
            try:
                args[key] = int(argv[i + 1]); assert args[key] > 0
            except Exception:
                raise SystemExit(f"❌ Uso: {flag} N (entero positivo)")
            argv = argv[:i] + argv[i + 2:]

    def pop_float01(flag, key):
        nonlocal argv
        if flag in argv:
            i = argv.index(flag)
            try:
                val = float(argv[i + 1])
                if not (0.0 <= val <= 1.0): raise ValueError
                args[key] = val
            except Exception:
                raise SystemExit(f"❌ Uso: {flag} C (float en [0,1])")
            argv = argv[:i] + argv[i + 2:]

    pop_int("--ventana_k", "ventana_k")
    pop_int("--meses", "meses")
    pop_int("--pool_min", "pool_min")
    pop_float01("--min_coverage", "min_coverage")

    if "--estricto" in argv:
        args["estricto"] = True
        argv = [a for a in argv if a != "--estricto"]

    if argv == ["--todo"]:
        args["todo"] = True
        args["decada"] = "TODO"
        return args

    if len(argv) == 2 and argv[0] == "--decada" and re.fullmatch(r"\d{4}_\d{4}", argv[1]):
        args["decada"] = argv[1]
        return args

    raise SystemExit(
        "Uso permitido:\n"
        "  py 05_compute_novelty.py --todo [--ventana_k K] [--meses M] [--estricto] [--pool_min N] [--min_coverage C]\n"
        "  py 05_compute_novelty.py --decada 1960_1969 [--ventana_k K] [--meses M] [--estricto] [--pool_min N] [--min_coverage C]\n"
    )

args = parse_args_strict()
DECADA       = args["decada"]
VENTANA_K    = args["ventana_k"]    # int|None
MESES        = args["meses"]        # int|None
ESTRICTO     = args["estricto"]     # bool
POOL_MIN     = args["pool_min"]     # int|None
MIN_COVERAGE = args["min_coverage"] # float|None

# -------------------------------
# Rutas de entrada/salida
# -------------------------------
PERCENTILES = [0, 5, 10, 20]

def _suffix():
    parts = [f"p{'_'.join(str(p) for p in PERCENTILES)}"]
    if VENTANA_K is not None: parts.append(f"k{VENTANA_K}")
    if MESES is not None:     parts.append(f"m{MESES}")
    if ESTRICTO:              parts.append("strict")
    if POOL_MIN is not None:  parts.append(f"pool{POOL_MIN}")
    if MIN_COVERAGE is not None: parts.append(f"cov{MIN_COVERAGE}")
    return "_".join(parts)

INPUT_PATH = os.path.join(TMP_DIR, f"04_vectores_{DECADA}.pkl")
CSV_OUTPUT = os.path.join(TMP_DIR, f"05_novelty_scores_{DECADA}_{_suffix()}.csv")
PKL_OUTPUT = os.path.join(TMP_DIR, f"05_novelty_scores_{DECADA}_{_suffix()}.pkl")

# -------------------------------
# Utilidades internas
# -------------------------------
def _period_to_int(p: pd.Period) -> int:
    # p.year*12 + p.month (asumiendo p no-nulo); si NaT, devolvemos un sentinel bajo
    if isinstance(p, pd.Period):
        return p.year * 12 + p.month
    return -10**9

def _valid_prev_indices_fast(i, period_nums, usable, meses=None, k=None, estricto=False, pool_min=None):
    """
    Devuelve índices j < i que cumplen:
      - j en [start, end) determinado vía búsqueda binaria (searchsorted):
            start por 'meses' (si aplica), end según 'estricto'
      - usable[j] (vector no nulo y (opcional) cobertura >= MIN_COVERAGE)
      - Si 'estricto' y pool < pool_min, relaja end para incluir coetáneos del mismo mes (<= current)
      - Si 'k' se especifica, limita a los K más recientes
    """
    if i == 0:
        return np.empty(0, dtype=np.int64)

    current = period_nums[i]

    # Límite inferior por ventana de meses (binary search)
    if meses is not None:
        lo = current - meses
        start = int(np.searchsorted(period_nums, lo, side="left"))
        start = min(start, i)  # no mirar más allá de i
    else:
        start = 0

    # Límite superior según 'estricto'
    if estricto:
        # < current (excluye coetáneos): primer índice con >= current en [0..i)
        end = int(np.searchsorted(period_nums, current, side="left", sorter=None))
        end = min(end, i)
    else:
        # <= current: todo hasta i
        end = i

    if start >= end:
        return np.empty(0, dtype=np.int64)

    # Filtrado por 'usable' (cobertura y vector no-nulo)
    # devolvemos indices absolutos
    slice_mask = usable[start:end]
    if not slice_mask.any():
        # fallback si estricto+pool_min y no se alcanza
        if estricto and (pool_min is not None):
            end2 = i  # incluir coetáneos
            if start >= end2:
                return np.empty(0, dtype=np.int64)
            slice_mask2 = usable[start:end2]
            idx2 = np.nonzero(slice_mask2)[0] + start
            if idx2.size == 0:
                return idx2
            # limitar por K (últimos k más recientes)
            if (k is not None) and (idx2.size > k):
                idx2 = idx2[-k:]
            return idx2
        return np.empty(0, dtype=np.int64)

    idx = np.nonzero(slice_mask)[0] + start

    # respaldo si estricto y pool_min especificado y no se alcanza
    if estricto and (pool_min is not None) and (idx.size < pool_min):
        end2 = i  # incluir coetáneos del mismo mes
        slice_mask2 = usable[start:end2]
        idx = np.nonzero(slice_mask2)[0] + start

    # limitar por K (últimos k más recientes)
    if (k is not None) and (idx.size > k):
        idx = idx[-k:]

    return idx

# -------------------------------
# Cálculo de novedad (rápido)
# -------------------------------
def compute_novelty(df: pd.DataFrame, percentiles, meses=None, k=None, estricto=False, pool_min=None, min_cov=None):
    """
    Para cada documento i:
      - Selecciona previos j < i (rápido con searchsorted) según filtros y 'usable' (vector != 0 y cobertura >= min_cov si aplica)
      - Calcula distancias coseno (1 - cos_sim) contra V[i] (normalizado)
      - Novedad = percentiles(dists)
      - Devuelve DF con novelty_pX + pool_size
    """
    # Periodos y orden
    df["anio_mes"] = pd.to_datetime(df["cover_date"], errors="coerce").dt.to_period("M")
    period_nums = df["anio_mes"].apply(_period_to_int).to_numpy()

    # Matriz de vectores (float32) y normalización por fila
    V = np.vstack(df["vector"].values).astype(np.float32, copy=False)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    nonzero_mask = (norms[:, 0] > 0.0)
    norms = np.clip(norms, 1e-12, None)
    V = V / norms

    # Máscara 'usable' (vector no nulo & cobertura >= umbral si existe)
    if (min_cov is not None) and ("coverage_ratio" in df.columns):
        cov_ok = (df["coverage_ratio"].fillna(0.0).to_numpy() >= float(min_cov))
        usable = nonzero_mask & cov_ok
    else:
        usable = nonzero_mask

    results = []
    it = tqdm(range(len(df)), desc=f"🔍 Novedad (K={k or 'ALL'}, M={meses or 'ALL'}, strict={estricto}, pool_min={pool_min or '—'}, min_cov={min_cov if min_cov is not None else '—'})")
    for i in it:
        if not usable[i]:
            novelty = {f"novelty_p{p}": np.nan for p in percentiles}
            pool_size = 0
        else:
            prev_idx = _valid_prev_indices_fast(
                i, period_nums, usable,
                meses=meses, k=k, estricto=estricto, pool_min=pool_min
            )
            pool_size = int(prev_idx.size)
            if pool_size == 0:
                novelty = {f"novelty_p{p}": np.nan for p in percentiles}
            else:
                sims = V[prev_idx] @ V[i]          # cos_sim
                dists = 1.0 - sims                 # 1 - cos
                novelty = {f"novelty_p{p}": float(np.percentile(dists, p)) for p in percentiles}

        novelty["eid"] = df.at[i, "eid"]
        novelty["pool_size"] = pool_size
        results.append(novelty)

    return pd.DataFrame(results)

# -------------------------------
# Principal
# -------------------------------
def main():
    t0 = time.time()

    print("📦 Cargando vectores de documentos...")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"❌ No existe {INPUT_PATH}. Ejecuta 04_vectorize_documents.py primero.")
    with open(INPUT_PATH, "rb") as f:
        df = pickle.load(f)

    # Validaciones
    needed = {"eid", "cover_date", "vector"}
    if not needed.issubset(df.columns):
        raise ValueError(f"❌ Faltan columnas requeridas: {needed}")

    print("📆 Asegurando datetime y orden estable (cover_date, eid)...")
    df["cover_date"] = pd.to_datetime(df["cover_date"], errors="coerce")
    sort_cols = [c for c in ["cover_date", "eid"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Info previa
    total_docs = len(df)
    zero_docs = int((df["vector"].apply(lambda v: isinstance(v, np.ndarray) and np.allclose(v, 0.0))).sum())
    cov_info = ""
    if "coverage_ratio" in df.columns:
        cov_mean = float(df["coverage_ratio"].mean())
        cov_info = f" | cobertura_media={cov_mean:.3f}"
    print(f"ℹ️ Documentos: {total_docs} | con vector nulo: {zero_docs} ({zero_docs/total_docs:.2%}){cov_info}")
    print(f"⚙️ Parámetros: percentiles={PERCENTILES}, ventana_k={VENTANA_K or 'ALL'}, meses={MESES or 'ALL'}, estricto={ESTRICTO}, pool_min={POOL_MIN or '—'}, min_coverage={MIN_COVERAGE if MIN_COVERAGE is not None else '—'}")

    print("🧠 Calculando índices de novedad (método optimizado)...")
    novelty_df = compute_novelty(
        df, PERCENTILES,
        meses=MESES, k=VENTANA_K,
        estricto=ESTRICTO, pool_min=POOL_MIN,
        min_cov=MIN_COVERAGE
    )

    print("📊 Resumen de novedad por percentil:")
    for p in PERCENTILES:
        col = f"novelty_p{p}"
        s = novelty_df[col].describe()
        if s["count"] > 0:
            print(f"  p{p}: count={int(s['count'])} mean={s['mean']:.4f} min={s['min']:.4f} max={s['max']:.4f}")
        else:
            print(f"  p{p}: sin datos (demasiados NaN, revisa filtros)")

    print("💾 Guardando resultados...")
    os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
    meta_cols = [c for c in ["eid", "doi", "citedby_count", "title", "cover_date"] if c in df.columns]
    final_df = df[meta_cols].merge(novelty_df, on="eid", how="left")

    final_df.to_csv(CSV_OUTPUT, index=False)
    with open(PKL_OUTPUT, "wb") as f:
        pickle.dump(final_df, f)

    elapsed = time.time() - t0
    print(f"\n✅ Resultados guardados en:\n- {CSV_OUTPUT}\n- {PKL_OUTPUT}")
    print(f"⏱️ Tiempo total: {elapsed/60:.2f} min")

if __name__ == "__main__":
    main()
