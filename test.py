import os
import json
import numpy as np
from tqdm import tqdm
import torch
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
CHUNKS_FILE = "/all_chunk_qcdt.json"   
MODEL_NAME = os.getenv("EMBED_MODEL", "AITeamVN/Vietnamese_Embedding")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE = os.getenv("DATABASE", "soni_agent")
COLLECTION = os.getenv("COLLECTION", "docs")
INITIAL_BATCH = int(os.getenv("BATCH_SIZE", 32))
MIN_BATCH = 1
BULK_INSERT_BATCH = 512
# ----------------------------

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)
if not isinstance(chunks, list):
    raise SystemExit("ERROR: expected a list of strings in the JSON file.")
print(f"Loaded {len(chunks)} items from {CHUNKS_FILE}")

print("Loading embedder:", MODEL_NAME)
embedder = SentenceTransformer(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    embedder.to(device)
except Exception:
    pass
embed_dim = embedder.get_sentence_embedding_dimension()
print("Device:", device, "embedding dim:", embed_dim)

def compute_embeddings_safe(texts, init_batch=INITIAL_BATCH, min_batch=MIN_BATCH):
    batch = init_batch
    all_emb = []
    i = 0
    total = len(texts)
    pbar = tqdm(total=total, desc="Embedding")
    while i < total:
        try:
            j = min(i + batch, total)
            cur = texts[i:j]
            emb = embedder.encode(cur, batch_size=batch, convert_to_numpy=True, show_progress_bar=False)
            all_emb.append(emb)
            pbar.update(len(cur))
            i = j
        except RuntimeError as e:
            print("RuntimeError during encode:", str(e))
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            if batch <= min_batch:
                pbar.close()
                raise RuntimeError("Batch is minimal and still failing") from e
            batch = max(min_batch, batch // 2)
            print(f"Reducing batch size to {batch} and retrying...")
    pbar.close()
    if len(all_emb) == 0:
        return np.zeros((0, embed_dim))
    return np.vstack(all_emb)

test_n = min(8, len(chunks))
print("Testing embeddings on a small subset...")
test_emb = compute_embeddings_safe(chunks[:test_n], init_batch=min(INITIAL_BATCH, test_n))
print("Test embeddings shape:", test_emb.shape)

print("Computing embeddings for all items...")
embeddings = compute_embeddings_safe(chunks, init_batch=INITIAL_BATCH)
print("Computed embeddings shape:", embeddings.shape)

client = MongoClient(MONGODB_URI)
db = client[DATABASE]
col = db[COLLECTION]


docs = []
count = 0
for i, (text, emb) in enumerate(zip(chunks, embeddings)):
    doc = {
        "text": text,
        "embedding": emb.tolist(),
        "meta": {"source": "qcdt", "index": i}
    }
    docs.append(doc)
    if len(docs) >= BULK_INSERT_BATCH:
        col.insert_many(docs)
        count += len(docs)
        docs = []
if docs:
    col.insert_many(docs)
    count += len(docs)

print(f"Inserted {count} documents into {DATABASE}.{COLLECTION}")
client.close()
print("Done.")
