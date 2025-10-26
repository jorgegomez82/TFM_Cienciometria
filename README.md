# 📊 TFM Cienciometría
**Índice de novedad en publicaciones científicas del campo de las matemáticas**

Este proyecto forma parte del Trabajo de Fin de Máster y tiene como objetivo **estimar un índice de novedad en artículos científicos del área de matemáticas**, a partir del análisis semántico de títulos y resúmenes.  
Se utilizan enfoques de **Word2Vec** y **MathBERT**, junto con técnicas de cienciometría.

---

## 📂 Estructura del repositorio

```
TFM_Cienciometria/
├── logs/                  # Contiene los logs del proyecto
│            
├── scripts/               # Scripts de procesamiento
│   ├── embeddings/
│   │   ├── word2Vec/      # Scripts para pipeline Word2Vec
│   │   ├── mathbert/      # Notebook específico de MathBERT
│   │   │   └── 04_mathbert_vectorizar.ipynb   # Se ejecuta en Colab
│   │   ├── 05_computar_novedad_word2vec_mathbert.py  # Se ejecuta en Colab
│   │   └── 06_visualizar_novedad_word2vec_mathbert.ipynb # Se ejecuta en Colab
│   └── recoleccionDatosAPI/ # Descarga desde API de Scopus
├── README.md
├── requirements.txt
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
- **`05_computar_disrupcion_word2vec_mathbert.ipynb`**
- **`06_visualizar_disrupción_word2vec_mathbert.ipynb`**

---

## 💾 Datos en Zenodo
Para cumplir con los términos de uso de la API de Scopus, este repositorio no contiene los metadatos brutos de los artículos. En su lugar, se publica en **Zenodo** el conjunto de datos oficial del proyecto, que consiste en la lista completa de identificadores (EIDs) de Scopus utilizados.

Publicar los EIDs garantiza la **total reproducibilidad** de la investigación, permitiendo que cualquier persona con acceso a Scopus pueda reconstruir el corpus original.

El dataset cuenta con un **DOI (Digital Object Identifier)**, lo que lo convierte en un recurso citable y permanente.

➡️ **Accede y cita el conjunto de datos aquí:**

[![DOI]](https://doi.org/10.5281/zenodo.17445712)

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