"""
03_train_word2vec.py — coherente con 01/02 (solo --todo / --decada)

- --todo  : entrena un modelo Word2Vec GLOBAL con los tokens generados por 02.
- --decada: si existe el modelo global, NO reentrena (para mantener comparabilidad);
            si NO existe, entrena un modelo LOCAL para esa década.

Entrada (según BASE = TODO o {YYYY_YYYY}):
  - Preferente: 02_preprocesado_{BASE}.pkl
  - Fallback  : 02_preprocesado_filtrado_{BASE}.pkl  (por si en el futuro existe)

Salidas:
  - 03_word2vec_{BASE}.model  (BASE=TODO o {DECADA})
  - 03_word2vec_{BASE}.kv
  - 03_word2vec_log_{BASE}.json
  - imagenes/03_normas_vectoriales_{BASE}.png
  - 03_word2vec_epoch_history_{BASE}.csv
"""

import os
import sys
import re
import time
import json
import pickle
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

from embeddings.config_paths import TMP_DIR

# ==========================================================
# CLI estricta (solo 2 formas)
# ==========================================================
USING_COMMON = True
try:
    from tfm_common import parse_args_strict  # type: ignore
except Exception:
    USING_COMMON = False

    def parse_args_strict():
        argv = sys.argv[1:]
        if argv == ["--todo"] or argv == ["--", "todo"]:
            return {"todo": True, "decada": None}
        if len(argv) == 2 and argv[0] == "--decada" and re.fullmatch(r"\d{4}_\d{4}", argv[1]):
            return {"todo": False, "decada": argv[1]}
        raise SystemExit(
            "Uso permitido (solo 2 formas):\n"
            "  1) py 03_train_word2vec.py --todo\n"
            "  2) py 03_train_word2vec.py --decada 1960_1969\n"
        )

# ==========================================================
# Parámetros de entrenamiento
# ==========================================================
VECTOR_SIZE = 200
WINDOW = 10
MIN_COUNT = 2      # coherente con bigramas: poda rarezas y reduce RAM/tiempo
SG = 1             # 1 = skip-gram, 0 = CBOW
EPOCHS = 12
WORKERS = max(1, min((os.cpu_count() or 2) - 2, 6))  # como máximo 6

# ==========================================================
# Callback de progreso + HISTÓRICO por época
# ==========================================================
@dataclass
class EpochStat:
    epoch: int
    seconds_this_epoch: float
    seconds_total: float
    tokens_per_sec: float
    loss_total: float
    loss_delta: float
    loss_delta_pct: float

class TqdmEpochProgress(CallbackAny2Vec):
    def __init__(self, total_epochs: int, base_tag: str, csv_dir: str):
        self.total = total_epochs
        self.base_tag = base_tag
        self.csv_dir = csv_dir
        self.pbar = None
        self.epoch = 0
        self.t0 = None
        self.last = None
        self.last_loss = None
        self.tokens_per_epoch = None
        self.history = []

    def on_train_begin(self, model):
        self.pbar = tqdm(total=self.total, desc="🧠 Entrenando (épocas)", unit="época")
        self.epoch = 0
        self.t0 = time.time()
        self.last = self.t0
        try:
            self.tokens_per_epoch = getattr(model, "corpus_total_words", None)
        except Exception:
            self.tokens_per_epoch = None

    def on_epoch_end(self, model):
        self.epoch += 1
        now = time.time()
        dt = now - self.last
        total = now - self.t0

        try:
            loss_total = float(model.get_latest_training_loss())
        except Exception:
            loss_total = float("nan")
        if self.last_loss is None or not np.isfinite(self.last_loss):
            loss_delta = float("nan")
            loss_delta_pct = float("nan")
        else:
            loss_delta = loss_total - self.last_loss
            loss_delta_pct = (loss_delta / self.last_loss * 100.0) if self.last_loss != 0 else float("nan")
        self.last_loss = loss_total

        if self.tokens_per_epoch and dt > 0:
            tps = self.tokens_per_epoch / dt
        else:
            tps = float("nan")

        postfix = (f"ep={self.epoch}/{self.total}  Δt={dt:.1f}s  total={total/60:.1f}m  "
                   f"{'' if np.isnan(tps) else f'~{tps:,.0f} tok/s  '}"
                   f"{'' if np.isnan(loss_total) else f'loss={int(loss_total)}  '}"
                   f"{'' if np.isnan(loss_delta) else f'Δloss={int(loss_delta)}  '}"
                   f"{'' if np.isnan(loss_delta_pct) else f'Δ%={loss_delta_pct:.2f}%'}")
        if self.pbar is not None:
            self.pbar.set_postfix_str(postfix)
            self.pbar.update(1)

        tqdm.write(
            f"[Época {self.epoch:>2}/{self.total}] Δt={dt:.1f}s | total={total/60:.1f}m | "
            f"{'' if np.isnan(tps) else f'~{tps:,.0f} tok/s | '}"
            f"{'' if np.isnan(loss_total) else f'loss={int(loss_total)} | '}"
            f"{'' if np.isnan(loss_delta) else f'Δloss={int(loss_delta)} | '}"
            f"{'' if np.isnan(loss_delta_pct) else f'Δ%={loss_delta_pct:.2f}%'}"
        )

        self.history.append(EpochStat(
            epoch=self.epoch,
            seconds_this_epoch=dt,
            seconds_total=total,
            tokens_per_sec=tps if np.isfinite(tps) else np.nan,
            loss_total=loss_total if np.isfinite(loss_total) else np.nan,
            loss_delta=loss_delta if np.isfinite(loss_delta) else np.nan,
            loss_delta_pct=loss_delta_pct if np.isfinite(loss_delta_pct) else np.nan
        ))

        self.last = now
        if self.epoch >= self.total and self.pbar is not None:
            self.pbar.close()

    def save_csv(self):
        os.makedirs(self.csv_dir, exist_ok=True)
        out_csv = os.path.join(self.csv_dir, f"03_word2vec_epoch_history_{self.base_tag}.csv")
        df_hist = pd.DataFrame([asdict(h) for h in self.history])
        df_hist.to_csv(out_csv, index=False)
        tqdm.write(f"📝 Historial de épocas guardado en: {out_csv}")

# ==========================================================
# Utilidades
# ==========================================================
def analyze_vocab(corpus, thresholds=[1, 2, 3, 5, 8, 10]):
    print("\n🔍 Análisis de frecuencia de vocabulario:")
    flat_tokens = (t for doc in corpus for t in doc)
    counts = Counter(flat_tokens)
    total = len(counts)
    print(f"🔢 Vocabulario total (sin filtrar): {total} palabras únicas")
    for thresh in thresholds:
        kept = sum(1 for _, c in counts.items() if c >= thresh)
        print(f"  - min_count={thresh:2}: se mantienen {kept:7} palabras ({total - kept} descartadas)")

def load_preprocessed(base: str):
    """
    Carga pickles generados por 02 (preferente) o, si existiera,
    la variante 'filtrado'. Devuelve DataFrame con columna 'tokens'.
    """
    p02  = os.path.join(TMP_DIR, f"02_preprocesado_{base}.pkl")
    p02f = os.path.join(TMP_DIR, f"02_preprocesado_filtrado_{base}.pkl")
    path = p02 if os.path.exists(p02) else p02f
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No se encontró {p02} ni {p02f}. Ejecuta 02 primero.")
    print(f"📦 Cargando tokens desde: {path}")
    with open(path, "rb") as f:
        df = pickle.load(f)
    if "tokens" not in df.columns or df["tokens"].isnull().all():
        raise ValueError("❌ La entrada no contiene una columna válida 'tokens'.")
    return df

def save_artifacts(model, base, elapsed):
    model_path = os.path.join(TMP_DIR, f"03_word2vec_{base}.model")
    kv_path    = os.path.join(TMP_DIR, f"03_word2vec_{base}.kv")
    log_path   = os.path.join(TMP_DIR, f"03_word2vec_log_{base}.json")
    img_path   = os.path.join(TMP_DIR, "imagenes", f"03_normas_vectoriales_{base}.png")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(img_path),   exist_ok=True)

    print("💾 Guardando modelo entrenado...")
    model.save(model_path)
    model.wv.save(kv_path)
    print(f"📁 Modelo completo: {model_path}")
    print(f"📁 Vectores Word2Vec (.kv): {kv_path}")

    norms = [np.linalg.norm(model.wv[w]) for w in model.wv.index_to_key]
    avg_norm = float(np.mean(norms)) if norms else 0.0
    print(f"📏 Norma promedio de los vectores de palabras: {avg_norm:.4f}")

    plt.figure(figsize=(10, 5))
    plt.hist(norms, bins=40, edgecolor="black")
    plt.title("Distribución de normas vectoriales de palabras")
    plt.xlabel("Norma del vector")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    print(f"📊 Histograma guardado en: {img_path}")

    mem_mb = psutil.Process().memory_info().rss / 1024 ** 2
    log_data = {
        "vector_size": VECTOR_SIZE,
        "window": WINDOW,
        "min_count": MIN_COUNT,
        "sg": SG,
        "epochs": EPOCHS,
        "workers": WORKERS,
        "vocab_size": len(model.wv),
        "avg_vector_norm": round(avg_norm, 4),
        "training_time_sec": round(elapsed, 2),
        "memory_used_mb": round(mem_mb, 2)
    }
    with open(log_path, "w") as logf:
        json.dump(log_data, logf, indent=4, ensure_ascii=False)
    print(f"📝 Log de entrenamiento guardado en: {log_path}")

def train_with_progress(corpus, base_tag: str):
    """
    Entrena Word2Vec en dos pasos con barra + histórico por época.
    """
    print("\n📚 Construyendo vocabulario...")
    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        workers=WORKERS,
        epochs=EPOCHS
    )
    model.build_vocab(corpus, progress_per=100_000)
    print(f"   ✔️ Palabras en vocab: {len(model.wv)}  |  docs: {len(corpus)}")

    print("⚙️ Entrenando (con barra + histórico por época)...")
    cb = TqdmEpochProgress(EPOCHS, base_tag=base_tag, csv_dir=TMP_DIR)
    t0 = time.time()
    model.train(
        corpus,
        total_examples=len(corpus),
        epochs=EPOCHS,
        compute_loss=True,
        callbacks=[cb]
    )
    elapsed = time.time() - t0
    cb.save_csv()
    return model, elapsed

# ==========================================================
# Main
# ==========================================================
def main():
    args = parse_args_strict()
    base = "TODO" if args["todo"] else args["decada"]

    # Ruta del modelo global para coherencia en modo década
    model_global_path = os.path.join(TMP_DIR, "03_word2vec_TODO.model")

    # Carga datos
    df = load_preprocessed(base)

    # Coherencia con 02: tokens ya bigramizados y filtrados
    corpus = df["tokens"].tolist()
    # Filtra documentos vacíos/mal formateados
    corpus = [doc for doc in corpus if isinstance(doc, (list, tuple)) and len(doc) > 0]
    print(f"✅ Corpus cargado con {len(corpus)} documentos (tras filtrar vacíos).")

    # Análisis previo del vocabulario (informativo)
    analyze_vocab(corpus)

    # Entrenamiento coherente con pipeline
    if args["todo"]:
        print(f"\n🧠 Entrenando modelo GLOBAL Word2Vec "
              f"(vector_size={VECTOR_SIZE}, sg={SG}, epochs={EPOCHS}, min_count={MIN_COUNT})...")
        model, elapsed = train_with_progress(corpus, base_tag="TODO")
        print(f"✅ Entrenamiento global completado en {elapsed:.2f} s.")
        save_artifacts(model, "TODO", elapsed)

    else:
        if os.path.exists(model_global_path):
            print("✔️ Modelo global encontrado: no se reentrena en modo década para mantener comparabilidad.")
            print(f"   Si necesitas entrenar por década, borra temporalmente:\n   {model_global_path}")
            return

        print(f"\n⚠️ No existe modelo global. Entrenando modelo LOCAL para {base} "
              f"(vector_size={VECTOR_SIZE}, sg={SG}, epochs={EPOCHS}, min_count={MIN_COUNT})...")
        model, elapsed = train_with_progress(corpus, base_tag=base)
        print(f"✅ Entrenamiento local ({base}) completado en {elapsed:.2f} s.")
        save_artifacts(model, base, elapsed)

if __name__ == "__main__":
    main()
