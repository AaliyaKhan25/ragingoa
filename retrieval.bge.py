import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time

# Load your saved dataset (raw passages — same corpus as before)
with open("dataset_subset.json", "r", encoding="utf-8") as f:
    records = json.load(f)

corpus = []
corpus_meta = []
for rec in records:
    for passage, is_sel in zip(rec["passages"], rec["is_selected"]):
        corpus.append(passage)
        corpus_meta.append({
            "query_id": rec["query_id"],
            "query": rec["query"],
            "is_selected": is_sel
        })

print(f"Total passages in corpus: {len(corpus)}")

# Load the stronger embedding model
print("Loading BGE model...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# BGE recommends NO prefix for passages, but a specific instruction prefix for queries
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

print("Embedding corpus...")
start = time.time()
corpus_embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
print(f"Embedding took {time.time() - start:.2f}s")

dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # inner product, since we normalized embeddings (cosine similarity)
index.add(corpus_embeddings.astype("float32"))
print(f"FAISS index built with {index.ntotal} vectors")

faiss.write_index(index, "faiss_index_bge.bin")
with open("corpus_meta.json", "w", encoding="utf-8") as f:
    json.dump({"corpus": corpus, "meta": corpus_meta}, f, ensure_ascii=False)

print("Saved faiss_index_bge.bin")

def search(query, k=5):
    query_with_prefix = QUERY_PREFIX + query
    query_emb = model.encode([query_with_prefix], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_emb, k)
    return [corpus_meta[idx] for idx in indices[0]]

# --- Evaluate on same 200 queries for fair comparison ---
K = 5
test_queries = records[:200]
hits = 0
latencies = []

for rec in test_queries:
    query = rec["query"]
    query_id = rec["query_id"]

    start = time.perf_counter()
    results = search(query, k=K)
    latency_ms = (time.perf_counter() - start) * 1000
    latencies.append(latency_ms)

    retrieved_for_this_query = [r for r in results if r["query_id"] == query_id]
    if any(r["is_selected"] == 1 for r in retrieved_for_this_query):
        hits += 1

recall_at_k = hits / len(test_queries)

print(f"\n=== BGE Retrieval Quality ===")
print(f"Recall@{K}: {recall_at_k:.2%} ({hits}/{len(test_queries)})")

print(f"\n=== BGE Latency (ms) across {len(test_queries)} queries ===")
print(f"P50:  {np.percentile(latencies, 50):.2f}ms")
print(f"P70:  {np.percentile(latencies, 70):.2f}ms")
print(f"P100: {np.percentile(latencies, 100):.2f}ms")
print(f"Mean: {np.mean(latencies):.2f}ms")