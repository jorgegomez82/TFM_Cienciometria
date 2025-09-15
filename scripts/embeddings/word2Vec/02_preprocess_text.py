# -*- coding: utf-8 -*-
"""
02_preprocess_text.py — Preprocesa título+abstract en tokens para Word2Vec:
- Aplica PhraseMatcher con lista blanca (prioriza la curada si existe).
- Tokeniza/lemma con spaCy (sin joblib; usa nlp.pipe con n_process).
- Entrena y aplica BIGRAMAS (Gensim Phrases/Phraser).
- Guarda PKL con 'tokens' ya bigramizados y el modelo .phraser.

CLI estricta (solo 2 formas):
  1) py 02_preprocess_text.py --todo
  2) py 02_preprocess_text.py --decada 1960_1969
"""
import os
import sys
import re
import csv
import pickle
from collections import Counter, defaultdict

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from tqdm import tqdm

from gensim.models.phrases import Phrases, Phraser

from embeddings.config_paths import TMP_DIR, CONFIG_DIR

# ==========================================================
# Fallback de utilidades comunes (CLI estricta, lectura, etc.)
# ==========================================================
USING_COMMON = True
try:
    from tfm_common import parse_args_strict  # type: ignore
    from tfm_common import IGNORED_COLS       # type: ignore
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
            "  1) py 02_preprocess_text.py --todo\n"
            "  2) py 02_preprocess_text.py --decada 1960_1969\n"
        )

    IGNORED_COLS = {"subject_areas", "num_subdisciplinas", "idioma"}

# =========================
# Configuración
# =========================
LISTA_BLANCA_CURADA = os.path.join(CONFIG_DIR, "lista_blanca_curada.csv")

N_JOBS = max(1, min((os.cpu_count() or 2) - 2, 6))  # como máximo 6
BATCH_SIZE = 1024  # tamaño de lote para nlp.pipe

# Limpieza de metadatos editoriales (antes de spaCy)
META_RX = (
    r"(?i)("
    r"copyright|©|all rights reserved|alle rechte vorbehalten|tous droits réservés|"
    r"reproduction|photocopying|microfilming|presses universitaires|"
    r"springer|verlag|wiley|elsevier|addison[- ]wesley|boca raton"
    r")"
)

# Filtrado de tokens meta (después de bigramas)
BAD_EXACT = {
    "copyright","all_rights","rights_reserved","alle_rechte",
    "rechte_vorbehalten","tous_droits","réservés","microfilm",
    "wiley","springer","elsevier","verlag","addison_wesley","boca_raton","press"
}
# para substrings seguros (si quieres mantener alguno):
BAD_SUBSTR_SAFE = ("wiley_vch",)  # opcional

def is_meta_token(tok: str) -> bool:
    t = tok.lower()
    if t in BAD_EXACT:
        return True
    return any(s in t for s in BAD_SUBSTR_SAFE)
# =========================
# Carga modelos y recursos
# =========================
if __name__ == "__main__":
    print("📦 Cargando spaCy en_core_web_sm ...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tok2vec"])

# Lista blanca opcional (solo multi-palabra, para unir con '_')
matcher = None
lista_blanca = set()
wl_path = LISTA_BLANCA_CURADA if os.path.exists(LISTA_BLANCA_CURADA) else None
if wl_path:
    if __name__ == "__main__":
        print(f"✅ Usando lista blanca: {os.path.basename(wl_path)}")
    df_wl = pd.read_csv(wl_path)
    if "palabra" in df_wl.columns:
        series = df_wl["palabra"]
    else:
        series = df_wl[df_wl.columns[0]]
    lista_blanca = set(series.astype(str).str.lower().str.strip())
    multi = [p for p in lista_blanca if " " in p or "_" in p]
    if multi:
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = []
        for p in multi:
            p_clean = p.replace("_", " ").strip()
            if p_clean:
                patterns.append(nlp.make_doc(p_clean))
        if patterns:
            matcher.add("WHITELIST_PHRASES", patterns)
else:
    if __name__ == "__main__":
        print("ℹ️ No se encontró lista blanca; se continuará sin PhraseMatcher.")

# =========================
# Reglas de descarte
# =========================
def _is_math_alnum_ok(s: str) -> bool:
    return bool(
        re.fullmatch(r"[a-z]\d{1,2}", s) or
        re.fullmatch(r"\d{1,2}[a-z]", s) or
        re.fullmatch(r"[a-z]{1,3}_\d{1,2}", s)
    )

def check_discard_reason(token):
    text = token.text
    low = text.lower()
    if (not token.is_alpha) and (low not in lista_blanca):
        if any(ch.isdigit() for ch in text) and _is_math_alnum_ok(low):
            pass
        else:
            return "no_alpha"
    if token.is_stop:
        return "stopword"
    if any(ch.isdigit() for ch in text) and (low not in lista_blanca) and (not _is_math_alnum_ok(low)):
        return "contiene_digito"
    if re.fullmatch(r"[a-z]{1,2}", low) and (low not in lista_blanca) and (not text.isupper()):
        return "sigla_corta"
    return None

# =========================
# Limpieza por documento
# =========================
def limpiar_doc(doc):
    """
    - Une frases de la lista blanca (si hay) con '_'.
    - Descarta tokens no deseados según reglas.
    - Devuelve (tokens_validos, motivos_descartes)
    """
    spans = []
    if matcher is not None:
        matches = matcher(doc)
        spans = [doc[s:e] for _, s, e in matches]
        spans = spacy.util.filter_spans(spans)

    tokens_validos = []
    motivos_descartes = defaultdict(Counter)
    indices_spans = {t.i for span in spans for t in span}

    for span in spans:
        tokens_validos.append(span.text.lower().replace(" ", "_"))

    for token in doc:
        if token.i in indices_spans:
            continue
        motivo = check_discard_reason(token)
        if (motivo is None) or (token.text.lower() in lista_blanca):
            lemma = token.lemma_.lower().strip()
            if lemma:
                tokens_validos.append(lemma)
        else:
            motivos_descartes[motivo][token.text] += 1

    return tokens_validos, motivos_descartes

# =========================
# Procesamiento en lote
# =========================
def procesar_con_spacy(textos, n_jobs=N_JOBS, batch_size=BATCH_SIZE):
    print(f"⚙️ spaCy nlp.pipe → n_process={n_jobs}, batch_size={batch_size}")
    tokens_validos = []
    discard_log = defaultdict(Counter)
    for doc in tqdm(nlp.pipe(textos, n_process=n_jobs, batch_size=batch_size), total=len(textos),
                    desc="📊 Preprocesando", unit="docs"):
        tv, md = limpiar_doc(doc)
        tokens_validos.append(tv)
        for motivo, counter in md.items():
            discard_log[motivo].update(counter)
    return tokens_validos, discard_log

def export_discard_log(discard_log, csv_log_path, csv_summary_path):
    all_discards = []
    for reason, counter in discard_log.items():
        for word, count in counter.items():
            all_discards.append((word, reason, count))
    if all_discards:
        df_descartes = pd.DataFrame(all_discards, columns=["palabra", "motivo", "frecuencia"])
        df_descartes = df_descartes.sort_values(by="frecuencia", ascending=False)
    else:
        df_descartes = pd.DataFrame(columns=["palabra", "motivo", "frecuencia"])
    resumen = (df_descartes.groupby("motivo")["frecuencia"].sum()
               .reset_index().sort_values("frecuencia", ascending=False))
    df_descartes.to_csv(csv_log_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    resumen.to_csv(csv_summary_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

# =========================
# Flujo principal
# =========================
def main():
    args = parse_args_strict()
    base = "TODO" if args["todo"] else args["decada"]

    # Entradas/salidas coherentes con 01_filter_language.py
    in_csv   = os.path.join(TMP_DIR, f"01_filtrado_abstract_ingles_{base}.csv")
    out_pkl  = os.path.join(TMP_DIR, f"02_preprocesado_{base}.pkl")
    csv_log  = os.path.join(TMP_DIR, f"02_log_palabras_descartadas_{base}.csv")
    csv_sum  = os.path.join(TMP_DIR, f"02_resumen_motivos_descartes_{base}.csv")
    bi_model = os.path.join(TMP_DIR, f"02_bigram_phraser_{base}.phraser")

    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"❌ No se encontró el archivo de entrada: {in_csv}")

    print(f"📥 Cargando {in_csv} ...")
    df = pd.read_csv(in_csv, engine="c", encoding="utf-8", on_bad_lines="skip", low_memory=False)
    df = df.drop(columns=list(IGNORED_COLS), errors="ignore")

    # Asegura 'text' y limpieza de metadatos editoriales
    if "text" not in df.columns:
        print("ℹ️ 'text' no existe; se compone a partir de 'title' y 'abstract'.")
        df["title"] = df.get("title", "").astype(str).fillna("").str.strip()
        df["abstract"] = df.get("abstract", "").astype(str).fillna("").str.strip()
        df["text"] = (df["title"] + ". " + df["abstract"]).str.strip()
    else:
        df["text"] = df["text"].astype(str).fillna("").str.strip()

    # 🔧 Limpieza de metadatos editoriales (multilingües)
    df["text"] = df["text"].str.replace(META_RX, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

    # Orden temporal (opcional)
    if "cover_date" in df.columns:
        df["cover_date"] = pd.to_datetime(df["cover_date"], errors="coerce")
        df = df.dropna(subset=["cover_date"]).sort_values(["cover_date", "eid"]).reset_index(drop=True)

    # =========================
    # 1) spaCy + PhraseMatcher → tokens base
    # =========================
    textos = df["text"].tolist()
    tokens_validos, discard_log = procesar_con_spacy(textos, n_jobs=N_JOBS, batch_size=BATCH_SIZE)

    # =========================
    # 2) BIGRAMAS (entrenar y aplicar)
    # =========================
    print("🧱 Entrenando BIGRAMAS (Gensim Phrases) sobre tokens...")
    phrases = Phrases(tokens_validos, min_count=30, threshold=60, progress_per=100_000)
    bigram = Phraser(phrases)
    bigram.save(bi_model)
    print(f"✅ Bigram Phraser guardado en: {bi_model}")

    print("🔗 Aplicando BIGRAMAS a los tokens...")
    tokens_bi = [bigram[toks] for toks in tqdm(tokens_validos, total=len(tokens_validos), desc="Aplicando bigramas")]

    # 🔧 Filtrado de tokens “meta” (post-bigramas)
    tokens_bi = [[t for t in doc if not is_meta_token(t)] for doc in tokens_bi]

    # =========================
    # Guardar resultados
    # =========================
    df["tokens"] = tokens_bi

    with open(out_pkl, "wb") as f:
        pickle.dump(df, f)
    print(f"✅ Guardado preprocesado: {out_pkl}")

    export_discard_log(discard_log, csv_log, csv_sum)
    print(f"📝 Log de descartes: {csv_log}")
    print(f"📝 Resumen de motivos: {csv_sum}")

if __name__ == "__main__":
    main()
