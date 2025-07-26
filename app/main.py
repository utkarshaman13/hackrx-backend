from fastapi import FastAPI, Request
from .embedding import build_faiss_index

app = FastAPI()

@app.post("/api/v1/hackrx/run")
async def run_query(request: Request):
    body = await request.json()
    text_chunks = body.get("chunks", [])

    index, chunk_refs = build_faiss_index(text_chunks)
    return {"message": "Index built successfully", "num_chunks": len(chunk_refs)}
