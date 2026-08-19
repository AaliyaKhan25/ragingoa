import json
import time
from contextlib import asynccontextmanager

import numpy as np
import faiss
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# --- Global state, loaded once at startup ---
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model, FAISS index, and corpus...")
    state["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    state["index"] = faiss.read_index("faiss_index.bin")

    with open("corpus_meta.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    state["corpus"] = data["corpus"]
    state["corpus_meta"] = data["meta"]

    # Warm-up call so the first real request isn't slow
    _ = state["model"].encode(["warm up query"], convert_to_numpy=True)
    print("Startup complete. Ready to serve.")

    yield  # app runs here

    state.clear()

app = FastAPI(title="Retrieval Service", lifespan=lifespan)


# --- Request/response schemas (structured I/O) ---
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

    query_emb = state["model"].encode([req.query], convert_to_numpy=True).astype("float32")
    distances, indices = state["index"].search(query_emb, req.k)

    chunks = []
    for idx, dist in zip(indices[0], distances[0]):
        chunks.append(RetrievedChunk(
            passage=state["corpus"][idx],
            score=float(dist)
        ))

    latency_ms = (time.perf_counter() - start) * 1000

    return RetrieveResponse(
        query=req.query,
        chunks=chunks,
        latency_ms=latency_ms
    )