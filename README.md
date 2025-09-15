# 📊 TFM Cienciometría
**Índice de novedad en publicaciones científicas del campo de las matemáticas**

Este proyecto forma parte del Trabajo de Fin de Máster y tiene como objetivo **estimar un índice de novedad en artículos científicos del área de matemáticas**, a partir del análisis semántico de títulos y resúmenes.  
Se utilizan enfoques de **Word2Vec** y **MathBERT**, junto con técnicas de cienciometría.

---

## 📂 Estructura del repositorio

```
TFM_Cienciometria/
├── data/                  # Conjuntos de datos (NO versionados en GitHub, se guardan en Zenodo)
│   ├── raw/               # Descargas iniciales desde la API de Scopus (por décadas)
│   ├── enriquecido/       # Datos enriquecidos con metadatos adicionales
│   ├── word2Vec/tmp/      # Resultados intermedios de Word2Vec
│   ├── mathbert/tmp/      # Resultados intermedios de MathBERT
│   └── tmp_articulos_abstract/
├── scripts/               # Scripts de procesamiento
│   ├── embeddings/
│   │   ├── word2Vec/      # Scripts para pipeline Word2Vec
│   │   ├── mathbert/      # Notebook específico de MathBERT
│   │   │   └── 04_mathbert_vectorizar.ipynb   # Se ejecuta en Colab
│   │   ├── 05_computar_novedad_word2vec_mathbert.ipynb  # Se ejecuta en Colab
│   │   └── 06_visualizar_novedad_word2vec_mathbert.ipynb # Se ejecuta en Colab
│   └── recoleccionDatosAPI/ # Descarga desde API de Scopus
├── README.md
├── requirements.txt
└── ignorar.gitignore
```

---

## ⚙️ Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/usuario/TFM_Cienciometria.git
   cd TFM_Cienciometria
   ```

2. Crea un entorno virtual e instala dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows

   pip install -r requirements.txt
   ```

3. **Modelos externos necesarios**:
   - Modelo de idioma FastText (`lid.176.ftz`) se descarga automáticamente en `/idioma_modelo/` cuando se ejecuta `01_filter_language.py`.
   - Modelo de spaCy:
     ```bash
     python -m spacy download en_core_web_sm
     ```

---

## 🚀 Pipeline de procesamiento

### 1. Recolección de datos desde Scopus
Los scripts en `/scripts/recoleccionDatosAPI/` descargan y enriquecen los artículos:
```bash
python scripts/recoleccionDatosAPI/searchAPI/main_06.py
python scripts/recoleccionDatosAPI/abstractAPI/enriquecer_desde_eid.py
```

### 2. Preprocesamiento de abstracts (Word2Vec)
```bash
cd scripts/embeddings/word2Vec

# Filtrado por idioma (usa FastText)
python 01_filter_language.py --todo

# Preprocesamiento y bigramas
python 02_preprocess_text.py --todo

# Entrenamiento Word2Vec
python 03_train_word2vec.py --todo

# Vectorización de documentos
python 04_vectorizar_word2vec.py --todo
```

### 3. Vectorización con MathBERT
En `/scripts/embeddings/mathbert/`, se ejecuta en Google Colab:
- **`04_mathbert_vectorizar.ipynb`**

### 4. Cálculo de novedad (Word2Vec + MathBERT)
Los dos notebooks compartidos para ambos enfoques (ejecutar en Colab):
- **`05_computar_novedad_word2vec_mathbert.ipynb`**
- **`06_visualizar_novedad_word2vec_mathbert.ipynb`**

---

## 💾 Datos en Zenodo
Debido al gran tamaño, los datasets no se incluyen en GitHub.  
Se almacenan en **Zenodo**, organizados en carpetas comprimidas:

- `raw/` → datos originales de Scopus (por décadas)
- `enriquecido/` → datos enriquecidos con metadatos
- `word2Vec/tmp/` → modelos entrenados y vectores
- `mathbert/tmp/` → embeddings y resultados intermedios

Cada carpeta en Zenodo incluye un archivo `README.txt` explicativo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## 📑 Requisitos principales
- `fasttext==0.9.2`
- `gensim==4.3.3`
- `spacy==3.8.7`
- `torch==2.5.0`
- `transformers==4.44.2`
- `pandas`, `numpy`, `tqdm`, `matplotlib`, `seaborn`

(Ver [requirements.txt](requirements.txt) para la lista completa).

---

## ✨ Créditos
Trabajo de Fin de Máster en Cienciometría.  
Autor: Jorge Gómez Gómez