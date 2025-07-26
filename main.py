from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Dummy embed function
def embed_text(text: str):
    # Pretend we call an embedding model and get back a list of floats
    return [0.1, 0.2, 0.3]

# Dummy build_index function
def build_faiss_index(texts: List[str]):
    # Pretend we build an index and return a success message
    return f"Index built with {len(texts)} items"

app = FastAPI(
    title="HackRx Backend",
    description="🚀 FastAPI backend with embedding and FAISS index endpoints",
    version="0.1.0"
)

class TextIn(BaseModel):
    text: str

class TextListIn(BaseModel):
    texts: List[str]

class EmbeddingOut(BaseModel):
    embedding: List[float]

class MessageOut(BaseModel):
    message: str

@app.get("/", response_model=MessageOut)
def read_root():
    """
    Root endpoint to check if the API is live.
    """
    return {"message": "Hello from FastAPI on Render!"}

@app.post("/embed", response_model=EmbeddingOut)
def embed_endpoint(item: TextIn):
    """
    Embed text and return vector.
    """
    vector = embed_text(item.text)
    return {"embedding": vector}

@app.post("/build-index", response_model=MessageOut)
def build_index_endpoint(item: TextListIn):
    """
    Build FAISS index from list of texts.
    """
    msg = build_faiss_index(item.texts)
    return {"message": msg}