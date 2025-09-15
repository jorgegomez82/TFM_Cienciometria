# audit_phraser.py
# Audita un modelo Gensim Phraser (bigramas):
# - Muestra nº de bigramas aprendidos y TOP-N por score.
# - Exporta todas las frases a CSV (bigram, score).
# - (Opcional) Aplica el phraser a una muestra de tu CSV para estimar cobertura.

import os
import csv
import argparse
from gensim.models.phrases import Phraser

def human_int(n):
    return f"{n:,}".replace(",", " ")

def load_phraser(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No existe el phraser: {path}")
    return Phraser.load(path)

def key_to_text(key):
    """Convierte la clave (tuple/list/bytes/str) en 'w1_w2_...'. """
    if isinstance(key, bytes):
        return key.decode("utf-8", "ignore")
    if isinstance(key, str):
        return key
    # Tupla/lista de partes (posible mezcla de bytes/str)
    parts = []
    for p in key:
        if isinstance(p, bytes):
            parts.append(p.decode("utf-8", "ignore"))
        else:
            parts.append(str(p))
    return "_".join(parts)

def print_top(bg, top=50):
    items = sorted(bg.phrasegrams.items(), key=lambda kv: -kv[1])[:top]
    if not items:
        print("ℹ️ No hay phrasegrams en el modelo.")
        return
    print(f"\n🏅 TOP {top} bigramas por score:")
    for key, score in items:
        print(f"  {key_to_text(key):<50} {score:.2f}")

def export_csv(bg, out_csv):
    rows = [(key_to_text(k), v) for k, v in bg.phrasegrams.items()]
    rows.sort(key=lambda x: -x[1])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bigram", "score"])
        w.writerows(rows)
    print(f"💾 Exportado CSV con {human_int(len(rows))} bigramas → {out_csv}")

def coverage_sample(bg, csv_path, text_col="text", nrows=50_000, encoding="utf-8"):
    import pandas as pd
    from tqdm import tqdm

    if not os.path.exists(csv_path):
        print(f"⚠️  Muestra omitida: no existe {csv_path}")
        return

    print(f"\n🔎 Estimando cobertura sobre muestra de {human_int(nrows)} docs de: {csv_path}")
    df = pd.read_csv(csv_path, usecols=[text_col], nrows=nrows, encoding=encoding)
    texts = df[text_col].astype(str).fillna("").tolist()

    orig_lens = new_lens = docs_changed = 0
    with tqdm(total=len(texts), desc="Aplicando phraser a muestra") as pbar:
        for t in texts:
            toks = t.split()
            toks_bi = bg[toks]
            orig_lens += len(toks)
            new_lens  += len(toks_bi)
            if len(toks_bi) != len(toks):
                docs_changed += 1
            pbar.update(1)

    delta = new_lens - orig_lens
    pct = (new_lens / orig_lens - 1.0) * 100 if orig_lens else 0.0
    print("\n📈 Cobertura estimada (tokenización simple por espacios):")
    print(f"  Tokens originales : {human_int(orig_lens)}")
    print(f"  Tokens con bigrama: {human_int(new_lens)}  (Δ {delta:+,} | {pct:+.2f}%)".replace(",", " "))
    print(f"  Docs con cambios  : {human_int(docs_changed)} / {human_int(len(texts))} "
          f"({docs_changed/len(texts)*100:.1f}%)")

def main():
    ap = argparse.ArgumentParser(description="Auditar modelo Gensim Phraser (bigramas).")
    ap.add_argument("--phraser", required=True, help="Ruta al archivo .phraser")
    ap.add_argument("--out_csv", default=None, help="Ruta para exportar (bigram,score).csv")
    ap.add_argument("--top", type=int, default=50, help="Cuántos bigramas mostrar en consola (por score)")
    ap.add_argument("--sample_csv", default=None, help="CSV para estimar cobertura (columna 'text' por defecto)")
    ap.add_argument("--text_col", default="text", help="Nombre de la columna de texto en sample_csv")
    ap.add_argument("--nrows", type=int, default=50000, help="Filas a leer de sample_csv para la estimación")
    ap.add_argument("--encoding", default="utf-8", help="Encoding de sample_csv (por defecto utf-8)")
    args = ap.parse_args()

    bg = load_phraser(args.phraser)
    print(f"✅ Phraser cargado: {args.phraser}")
    print(f"📦 Bigramas aprendidos (phrasegrams): {human_int(len(bg.phrasegrams))}")

    print_top(bg, top=args.top)

    out_csv = args.out_csv
    if out_csv is None:
        base_dir = os.path.dirname(args.phraser)
        base_name = os.path.splitext(os.path.basename(args.phraser))[0]
        out_csv = os.path.join(base_dir, f"{base_name}_frases.csv")
    export_csv(bg, out_csv)

    if args.sample_csv:
        coverage_sample(bg, args.sample_csv, text_col=args.text_col, nrows=args.nrows, encoding=args.encoding)

if __name__ == "__main__":
    main()
