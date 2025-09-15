import pickle

with open("../../data/word2Vec/tmp/preprocesado_1970_1979.pkl", "rb") as f:
    df_tokens = pickle.load(f)

num_docs = len(df_tokens)
total_tokens = sum(len(tokens) for tokens in df_tokens["tokens"])
unique_words = len(set(word for doc in df_tokens["tokens"] for word in doc))

print(f"N° documentos: {num_docs}")
print(f"N° total de palabras: {total_tokens}")
print(f"N° palabras únicas: {unique_words}")