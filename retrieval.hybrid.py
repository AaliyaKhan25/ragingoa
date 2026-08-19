import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
from rank_bm25 import BM25Okapi

# Load corpus + metadata (already built from your MiniLM run)
with open("corpus_meta.json", "r", encoding="utf-8") as f:
    data = json.load(f)
corpus = data["corpus"]
corpus_meta = data["meta"]

with open("dataset_subset.json", "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"Corpus size: {len(corpus)}")

# Load dense model + FAISS index (already built)
print("Loading MiniLM model + FAISS index...")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.bin")

# Build BM25 index (tokenize corpus — simple whitespace/lowercase split)
print("Building BM25 index...")
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
print("BM25 index built.")

def dense_search(query, k=20):
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_emb, k)
    return list(indices[0])  # ranked list of corpus indices

def bm25_search(query, k=20):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_k_idx = np.argsort(scores)[::-1][:k]
    return list(top_k_idx)

def hybrid_search(query, k=5, rrf_k=20, candidate_pool=40, dense_weight=0.7, bm25_weight=0.3):
    dense_ranked = dense_search(query, k=candidate_pool)
    bm25_ranked = bm25_search(query, k=candidate_pool)

    rrf_scores = {}
    for rank, idx in enumerate(dense_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + dense_weight * (1.0 / (rrf_k + rank + 1))
    for rank, idx in enumerate(bm25_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + bm25_weight * (1.0 / (rrf_k + rank + 1))

    fused_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [corpus_meta[idx] for idx, _ in fused_ranked]

    # Reciprocal Rank Fusion
    rrf_scores = {}
    for rank, idx in enumerate(dense_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(bm25_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

    # Sort by fused score, take top k
    fused_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [corpus_meta[idx] for idx, _ in fused_ranked]

# Warm-up (exclude first-call overhead from timing)
_ = hybrid_search(records[0]["query"], k=5)

# --- Evaluate on same 200 queries for fair comparison ---
K = 5
test_queries = records[:200]
hits = 0
latencies = []

for rec in test_queries:
    query = rec["query"]
    query_id = rec["query_id"]

    start = time.perf_counter()
    results = hybrid_search(query, k=K)
    latency_ms = (time.perf_counter() - start) * 1000
    latencies.append(latency_ms)

    retrieved_for_this_query = [r for r in results if r["query_id"] == query_id]
    if any(r["is_selected"] == 1 for r in retrieved_for_this_query):
        hits += 1

recall_at_k = hits / len(test_queries)

print(f"\n=== Hybrid (BM25 + MiniLM) Retrieval Quality ===")
print(f"Recall@{K}: {recall_at_k:.2%} ({hits}/{len(test_queries)})")

print(f"\n=== Hybrid Latency (ms) across {len(test_queries)} queries ===")
print(f"P50:  {np.percentile(latencies, 50):.2f}ms")
print(f"P70:  {np.percentile(latencies, 70):.2f}ms")
print(f"P100: {np.percentile(latencies, 100):.2f}ms")
print(f"Mean: {np.mean(latencies):.2f}ms")