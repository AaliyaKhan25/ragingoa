import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time

# Load saved index and corpus (fast — no re-embedding)
print("Loading model and index...")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.bin")

with open("corpus_meta.json", "r", encoding="utf-8") as f:
    data = json.load(f)
corpus = data["corpus"]
corpus_meta = data["meta"]

with open("dataset_subset.json", "r", encoding="utf-8") as f:
    records = json.load(f)

def search(query, k=5):
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_emb, k)
    return [corpus_meta[idx] for idx in indices[0]]

# --- Evaluate retrieval quality across many queries ---
K = 5
test_queries = records[:200]  # test on 200 queries
hits = 0
latencies = []

for rec in test_queries:
    query = rec["query"]
    query_id = rec["query_id"]

    start = time.perf_counter()
    results = search(query, k=K)
    latency_ms = (time.perf_counter() - start) * 1000
    latencies.append(latency_ms)

    # Check if any retrieved passage is the gold passage for THIS query
    retrieved_for_this_query = [r for r in results if r["query_id"] == query_id]
    if any(r["is_selected"] == 1 for r in retrieved_for_this_query):
        hits += 1

recall_at_k = hits / len(test_queries)

print(f"\n=== Retrieval Quality ===")
print(f"Recall@{K}: {recall_at_k:.2%} ({hits}/{len(test_queries)})")

print(f"\n=== Latency (ms) across {len(test_queries)} queries ===")
print(f"P50:  {np.percentile(latencies, 50):.2f}ms")
print(f"P70:  {np.percentile(latencies, 70):.2f}ms")
print(f"P100: {np.percentile(latencies, 100):.2f}ms")
print(f"Mean: {np.mean(latencies):.2f}ms")