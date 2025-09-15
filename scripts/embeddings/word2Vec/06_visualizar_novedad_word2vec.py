"""
06_visualize_novelty.py (simple y eficiente)

- Carga 05_novelty_scores_{DECADA}_*.pkl desde TMP_DIR (misma lógica de sufijo que 05).
- Exporta el TOP p20 (por percentil global, 99.9 por defecto).
- Grafica SOLO la línea de tendencia: mediana de citas (log1p) por bins de novelty_p20.
- Guarda también CSV con la curva de tendencia (x_bin_center, median_log_cites)

Uso:
  py 06_visualize_novelty.py --todo [--ventana_k K] [--meses M] [--estricto] [--pool_min N] [--min_coverage C] [--p20_pct 99.9]
  py 06_visualize_novelty.py --decada 1960_1969 [mismos opcionales]
"""

import os, re, glob, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from embeddings.config_paths import TMP_DIR

sns.set_style("whitegrid")

# ---------------- CLI (coherente con 05) ----------------
def parse_args_strict():
    argv = sys.argv[1:]
    args = {
        "todo": False, "decada": None,
        "ventana_k": None, "meses": None, "estricto": False, "pool_min": None,
        "min_coverage": None,
        "p20_pct": 99.9   # top 0.1% por defecto
    }

    def pop_val(flag, caster):
        nonlocal argv
        if flag in argv:
            i = argv.index(flag)
            try:
                val = caster(argv[i+1])
            except Exception:
                raise SystemExit(f"❌ Uso: {flag} <valor>")
            argv = argv[:i] + argv[i+2:]
            return val
        return None

    args["ventana_k"]    = pop_val("--ventana_k", int)
    args["meses"]        = pop_val("--meses", int)
    args["pool_min"]     = pop_val("--pool_min", int)
    args["min_coverage"] = pop_val("--min_coverage", float)
    _p = pop_val("--p20_pct", float)
    if _p is not None: args["p20_pct"] = _p

    if "--estricto" in argv:
        args["estricto"] = True
        argv = [a for a in argv if a != "--estricto"]

    if argv == ["--todo"]:
        args["todo"] = True; args["decada"] = "TODO"; return args
    if len(argv) == 2 and argv[0] == "--decada" and re.fullmatch(r"\d{4}_\d{4}", argv[1]):
        args["decada"] = argv[1]; return args

    raise SystemExit(
        "Uso permitido:\n"
        "  py 06_visualize_novelty.py --todo [--ventana_k K] [--meses M] [--estricto] [--pool_min N] "
        "[--min_coverage C] [--p20_pct 99.9]\n"
        "  py 06_visualize_novelty.py --decada 1960_1969 [mismos opcionales]\n"
    )

args = parse_args_strict()
DECADA       = args["decada"]
VENTANA_K    = args["ventana_k"]
MESES        = args["meses"]
ESTRICTO     = args["estricto"]
POOL_MIN     = args["pool_min"]
MIN_COVERAGE = args["min_coverage"]
P20_PCT      = args["p20_pct"]

# ---------------- Rutas (mismo sufijo que 05) ----------------
def _suffix():
    parts = ["p0_5_10_20"]
    if VENTANA_K is not None: parts.append(f"k{VENTANA_K}")
    if MESES is not None:     parts.append(f"m{MESES}")
    if ESTRICTO:              parts.append("strict")
    if POOL_MIN is not None:  parts.append(f"pool{POOL_MIN}")
    if MIN_COVERAGE is not None: parts.append(f"cov{MIN_COVERAGE}")
    return "_".join(parts)

def _expected_input_path():
    return os.path.join(TMP_DIR, f"05_novelty_scores_{DECADA}_{_suffix()}.pkl")

def _autodetect_input_path():
    pattern = os.path.join(TMP_DIR, f"05_novelty_scores_{DECADA}_*.pkl")
    cand = glob.glob(pattern)
    if not cand: return None
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]

def _resolve_input_path():
    target = _expected_input_path()
    if os.path.exists(target):
        print("📥 Archivo de entrada (exacto por flags):", target)
        return target
    print("ℹ️ No se encontró el archivo exacto; auto-detectando…")
    auto = _autodetect_input_path()
    if auto:
        print("📥 Archivo de entrada (auto):", auto)
        return auto
    raise FileNotFoundError("❌ No se encontró PKL 05_novelty_scores_; ejecuta 05 primero.")

INPUT_PATH = _resolve_input_path()
BASE_TAG   = os.path.splitext(os.path.basename(INPUT_PATH))[0]  # 05_novelty_scores_...

OUTPUT_DIR = TMP_DIR
IMG_DIR    = os.path.join(TMP_DIR, "imagenes")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ---------------- Utilidades ----------------
def _cols_export_fixed(df, score_col="novelty_p20"):
    order = ["eid", "cover_date", "citedby_count", score_col, "pool_size", "doi", "title"]
    return [c for c in order if c in df.columns]

# ---------------- Lógica principal ----------------
def main():
    print("📦 Cargando:", INPUT_PATH)
    df = pd.read_pickle(INPUT_PATH)

    # Tipos básicos
    df["cover_date"] = pd.to_datetime(df["cover_date"], errors="coerce")
    if "citedby_count" in df.columns:
        df["citedby_count"] = pd.to_numeric(df["citedby_count"], errors="coerce")

    # Convierte a numérico todas las columnas novelty_p* y (si existe) pool_size
    nov_cols = [col for col in df.columns if re.fullmatch(r"novelty_p\d+", str(col))]
    to_cast = nov_cols + (["pool_size"] if "pool_size" in df.columns else [])
    for c in to_cast:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---------- 1) Exportar TOP p20 por percentil ----------
    if "novelty_p20" not in df.columns:
        raise ValueError("❌ No existe columna 'novelty_p20' en el archivo de entrada.")
    thr = df["novelty_p20"].quantile(P20_PCT/100.0)
    top = df.loc[df["novelty_p20"] >= thr].copy().sort_values("novelty_p20", ascending=False)
    cols = _cols_export_fixed(top, "novelty_p20")
    out_top = os.path.join(OUTPUT_DIR, f"06_{BASE_TAG}_p20_top{P20_PCT:g}pct_{DECADA}.csv")
    top[cols].to_csv(out_top, index=False, encoding="utf-8")
    print(f"✅ TOP p20 ≥ P{P20_PCT:g}% guardado en: {out_top}   |   seleccionados={len(top):,}".replace(",", " "))

    # ---------- 2) Gráfico SOLO LÍNEA: mediana por bins de novelty_p20 ----------
    d = df[["novelty_p20", "citedby_count"]].dropna()
    if d.empty:
        print("⚠️ No hay datos suficientes para el gráfico.")
        return

    d["log_cites"] = np.log1p(d["citedby_count"])

    # Bins equidistantes en novelty_p20
    nbins = 60
    bins = np.linspace(d["novelty_p20"].min(), d["novelty_p20"].max(), nbins + 1)
    idx  = np.digitize(d["novelty_p20"], bins) - 1
    g = pd.DataFrame({"bin": idx, "y": d["log_cites"].values})
    trend = g.groupby("bin")["y"].median().reset_index()
    centers = (bins[:-1] + bins[1:]) / 2.0
    trend["x"] = centers[trend["bin"].clip(0, nbins-1)]
    trend = trend.dropna()

    # Plot de la línea (sin densidad)
    plt.figure(figsize=(9.5, 5.6))
    plt.plot(trend["x"], trend["y"], linewidth=2.5)
    plt.xlabel("Índice de novedad (novelty_p20 = 1 - cos)")
    plt.ylabel("Mediana de log1p(citedby_count)")
    plt.title(f"Citas (mediana log) vs. Novedad (percentil 20) — {DECADA}")
    plt.tight_layout()
    out_png = os.path.join(IMG_DIR, f"06_{BASE_TAG}_novelty_p20_vs_citations_LINE_{DECADA}.png")
    plt.savefig(out_png)
    plt.close()
    print(f"🖼️ Gráfico (solo línea) guardado en: {out_png}")

    # Guardar la curva
    trend_out = os.path.join(OUTPUT_DIR, f"06_{BASE_TAG}_trend_bins_{DECADA}.csv")
    trend.rename(columns={"y": "median_log_cites"}, inplace=True)
    trend[["x", "median_log_cites"]].to_csv(trend_out, index=False)
    print(f"📄 Curva de tendencia guardada en: {trend_out}")

if __name__ == "__main__":
    main()
