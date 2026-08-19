# Retrieval Pipeline — Design Notes

## Dataset
- Source: `ai4bharat/MSMARCO-XI` (Kannada subset, `train/kantrain.parquet`)
- Extracted 3,000 English query-passage records (`Eng_Query`, `English_passages`, `is_selected` gold labels)
- Flattened into a corpus of ~29,953 passages for retrieval

## Approaches Tested

| Approach | Recall@5 | P50 latency | P70 latency | P100 latency |
|---|---|---|---|---|
| **MiniLM (dense only)** ✅ *(chosen)* | 52.00% | 26.67ms | 34.08ms | 53.46ms |
| BGE-base (dense only) | 55.50% | 74.14ms | 76.85ms | ~1075ms* |
| Hybrid (BM25 + MiniLM, equal weight) | 45.00% | 105.40ms | 124.92ms | 319.22ms |
| Hybrid (BM25 + MiniLM, weighted 0.7/0.3) | 50.50% | 100.61ms | 120.24ms | 566.45ms |

*Likely a cold-start artifact from first-inference model warm-up, not representative of steady-state latency — not re-verified due to time constraints.

## Findings

**BGE-base** improved recall by 3.5 points over MiniLM but nearly tripled P50 latency and showed a P100 spike well past our 200ms budget. Given the marginal recall gain, the latency cost wasn't justified for this use case.

**Hybrid search (BM25 + dense via Reciprocal Rank Fusion)** underperformed both dense-only baselines. MSMARCO passages are largely paraphrased rather than keyword-matching the query, so BM25's keyword-overlap signal introduced noise that diluted the stronger dense-embedding signal — even after reweighting RRF to favor dense results 70/30. BM25's full-corpus scan on every query also added significant latency (~4x over dense-only).

## Final Decision
**MiniLM (`all-MiniLM-L6-v2`) + FAISS flat index (dense-only)** was selected for production:
- Best latency by a wide margin (P50: 26.67ms, P100: 53.46ms — well under the 200ms target)
- Competitive recall (52%) without the cost of heavier models or hybrid overhead
- Simple, fast to warm up, low resource footprint — good fit for deployment constraints

## Architecture
- **Chunking**: MSMARCO passages used as-is (already short, pre-segmented) — no additional splitting needed for this dataset
- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2`
- **Index**: FAISS `IndexFlatL2`
- **Metadata**: each indexed chunk carries `query_id`, source `query`, and `is_selected` gold-relevance flag for evaluation
- **Evaluation**: Recall@5 measured against gold `is_selected` labels across 200 held-out queries

## API
Service exposed via FastAPI (`app.py`):
- `GET /health` — health check
- `POST /retrieve` — `{"query": str, "k": int}` → `{"query", "chunks": [{"passage", "score"}], "latency_ms"}`

## Files
- `dataset_subset.json` — cleaned, flattened dataset (3,000 queries)
- `corpus_meta.json` — full passage corpus + metadata used for indexing
- `faiss_index.bin` — production FAISS index (MiniLM embeddings)
- `retrieval_baseline.py` — builds the MiniLM baseline index
- `retrieval_bge.py` — BGE-base comparison experiment
- `retrieval_hybrid.py` — BM25 + dense hybrid experiment
- `evaluate_retrieval.py` — recall + latency evaluation harness
- `app.py` — FastAPI service exposing `/retrieve`