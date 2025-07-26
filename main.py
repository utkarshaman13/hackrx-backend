from fastapi import FastAPI
from app.embedding import embed_text, build_faiss_index

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Render!"}

@app.post("/embed")
async def embed_endpoint(text: str):
    embedding = embed_text(text)
    return {"embedding": embedding}

@app.post("/build-index")
async def build_index_endpoint(texts: list[str]):
    index = build_faiss_index(texts)
    return {"index_built": True}
