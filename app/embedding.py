import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def embed_text(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

def build_faiss_index(chunks):
    import faiss
    import numpy as np

    vectors = [embed_text(chunk) for chunk in chunks]
    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors).astype('float32'))
    return index, list(range(len(chunks)))