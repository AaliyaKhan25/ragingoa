import pyarrow.parquet as pq
import pandas as pd
from huggingface_hub import hf_hub_download
import json

file_path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="train/kantrain.parquet",
    repo_type="dataset"
)

parquet_file = pq.ParquetFile(file_path)

SUBSET_SIZE = 3000
batches = parquet_file.iter_batches(batch_size=SUBSET_SIZE)
first_batch = next(batches)
df = first_batch.to_pandas()

print(f"Loaded {len(df)} rows")
print(df.columns.tolist())

records = []
for _, row in df.iterrows():
    passages_obj = row["passages"]

    if passages_obj is None:
        continue

    raw_passages = passages_obj.get("English_passages")
    raw_selected = passages_obj.get("is_selected")

    if raw_passages is None:
        clean_passages = []
    else:
        clean_passages = [str(x) for x in raw_passages]

    if raw_selected is None:
        clean_selected = []
    else:
        clean_selected = [int(x) for x in raw_selected]

    if len(clean_passages) == 0:
        continue

    if len(clean_selected) == 0:
        clean_selected = [0] * len(clean_passages)

    records.append({
        "query_id": int(row["query_id"]) if row["query_id"] is not None else None,
        "query": row["Eng_Query"],
        "answer": row["Eng_Answer"],
        "passages": clean_passages,
        "is_selected": clean_selected
    })

print(f"\nUsable records after cleaning: {len(records)}")
print("\n--- Sample record ---")
print(json.dumps(records[0], indent=2, ensure_ascii=False))

with open("dataset_subset.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("\nSaved to dataset_subset.json")