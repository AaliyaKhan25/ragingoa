import json
import os
import time
from contextlib import asynccontextmanager

import faiss
import numpy as np
import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- Global state ---
state = {}

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def get_hf_embedding(text: str) -> np.ndarray:
    """Fetch embeddings from Hugging Face API to bypass local PyTorch RAM usage."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    payload = {"inputs": [text], "options": {"wait_for_model": True}}

    response = requests.post(HF_API_URL, headers=headers, json=payload)
    response.raise_for_status()

    embedding = response.json()
    return np.array(embedding, dtype="float32")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading FAISS index and corpus...")
    state["index"] = faiss.read_index("faiss_index.bin")

    with open("corpus_meta.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    state["corpus"] = data["corpus"]
    state["corpus_meta"] = data["meta"]

    print("Startup complete. Ready to serve.")
    yield
    state.clear()


app = FastAPI(title="Retrieval Service", lifespan=lifespan)


# --- Request/response schemas ---
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    passage: str
    score: float


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    start = time.perf_counter()

    query_emb = get_hf_embedding(req.query)
    distances, indices = state["index"].search(query_emb, req.k)

    chunks = []
    for idx, dist in zip(indices[0], distances[0]):
        chunks.append(
            RetrievedChunk(
                passage=state["corpus"][idx], score=float(dist)
            )
        )

    latency_ms = (time.perf_counter() - start) * 1000

    return RetrieveResponse(
        query=req.query, chunks=chunks, latency_ms=latency_ms
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)