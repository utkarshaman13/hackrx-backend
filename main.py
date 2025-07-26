from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.embedding import embed_text, build_faiss_index

app = FastAPI()

class TextIn(BaseModel):
    text: str

class TextListIn(BaseModel):
    texts: List[str]

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Render!"}

@app.post("/embed")
def embed(text_in: TextIn):
    embedding = embed_text(text_in.text)
    return {"embedding": embedding}

@app.post("/build-index")
def build_index(texts_in: TextListIn):
    index = build_faiss_index(texts_in.texts)
    return {"message": "Index built successfully", "num_texts": len(texts_in.texts)}
