# scripts/config_paths.py
from pathlib import Path

# Raíz del repo = padre de /scripts
BASE_DIR = Path(__file__).resolve().parents[2]

# Árbol de datos
DATA_DIR        = BASE_DIR / "data"
ENRIQUECIDO_DIR = DATA_DIR / "enriquecido"

# Word2Vec
W2V_DIR    = DATA_DIR / "word2Vec"
TMP_DIR    = W2V_DIR / "tmp"
CONFIG_DIR = W2V_DIR / "lista_blanca_negra"

# MathBERT
MATHBERT_DIR     = DATA_DIR / "MathBERT"
MATHBERT_TMP = MATHBERT_DIR / "tmp"
MATHBERT_IMG = MATHBERT_DIR / "tmp/imagenes"

#Comparación de embeddings
COMPARACION_DIR = DATA_DIR / "comparacion_modelos"

# (Opcional) carpeta local de modelos de idioma si la usas
IDIOMA_DIR = BASE_DIR / "idioma_modelo"

# Crear si no existen
for d in [
    DATA_DIR, ENRIQUECIDO_DIR,
    W2V_DIR, TMP_DIR, CONFIG_DIR,
    MATHBERT_DIR, MATHBERT_TMP, MATHBERT_IMG,
    IDIOMA_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
