import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time

# Load your saved dataset
with open("dataset_subset.json", "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"Loaded {len(records)} records")

# Flatten ALL passages across all records into one corpus (this is what we'll search over)
corpus = []       # the actual passage text
corpus_meta = []  # metadata: which query it came from, whether it's the gold passage

for rec in records:
    for passage, is_sel in zip(rec["passages"], rec["is_selected"]):
        corpus.append(passage)
        corpus_meta.append({
            "query_id": rec["query_id"],
            "query": rec["query"],
            "is_selected": is_sel
        })

print(f"Total passages in corpus: {len(corpus)}")

# Load embedding model (small + fast for now)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed the whole corpus
print("Embedding corpus (this may take a minute)...")
start = time.time()
corpus_embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
print(f"Embedding took {time.time() - start:.2f}s")

# Build FAISS index
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings.astype("float32"))
print(f"FAISS index built with {index.ntotal} vectors")

# Save index + corpus for reuse
faiss.write_index(index, "faiss_index.bin")
with open("corpus_meta.json", "w", encoding="utf-8") as f:
    json.dump({"corpus": corpus, "meta": corpus_meta}, f, ensure_ascii=False)

print("Saved faiss_index.bin and corpus_meta.json")

# Quick test search
def search(query, k=5):
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_emb, k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "passage": corpus[idx],
            "distance": float(dist),
            "meta": corpus_meta[idx]
        })
    return results

# Test with the first query from your dataset
test_query = records[0]["query"]
print(f"\n--- Test query: {test_query} ---")
results = search(test_query, k=3)
for r in results:
    print(f"\nDistance: {r['distance']:.3f}")
    print(f"Passage: {r['passage'][:150]}...")
    print(f"Is gold passage: {r['meta']['is_selected']}")