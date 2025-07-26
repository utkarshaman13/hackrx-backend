from fastapi import FastAPI
from app.embedding import embed_text, build_faiss_index

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Render!"}