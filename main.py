from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.embedding import embed_text, build_faiss_index

app = FastAPI()

# Input model for embedding text
class TextIn(BaseModel):
    text: str

# Input model for building index
class TextListIn(BaseModel):
    texts: List[str]

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Render!"}

@app.post("/embed")
def embed(text_in: TextIn):
    """
    Generate embedding for a single text.
    """
    embedding = embed_text(text_in.text)
    return {"embedding": embedding}

@app.post("/build-index")
def build_index(texts_in: TextListIn):
    """
    Build a FAISS index from a list of texts.
    (In practice, you'd probably save and reuse this index, but for demo it just returns success.)
    """
    index = build_faiss_index(texts_in.texts)
    return {"message": "Index built successfully", "num_texts": len(texts_in.texts)}
