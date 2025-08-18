import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

# config từ env
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE = os.getenv("DATABASE", "soni_agent")
COLLECTION = os.getenv("COLLECTION", "docs")

MODEL_NAME = "AITeamVN/Vietnamese_Embedding" 
BATCH_SIZE = 64
CHUNKS_FILE = "all_chunk_qcdt.json"

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

embedder = SentenceTransformer(MODEL_NAME)

sample = embedder.encode(["test"], convert_to_numpy=True)
EMBED_DIM = sample.shape[1]
print("Embedding dim:", EMBED_DIM)

def compute_embeddings(texts, batch_size=BATCH_SIZE):
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        emb = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True, batch_size=batch_size)
        all_emb.append(emb)
    return np.vstack(all_emb)

embeddings = compute_embeddings(chunks)

client = MongoClient(MONGODB_URI)
db = client[DATABASE]
col = db[COLLECTION]


batch_docs = []
for i, (text, emb) in enumerate(zip(chunks, embeddings)):
    doc = {
        "text": text,
        "embedding": emb.tolist(),    
        "meta": {"source": "qcdt", "chunk_id": i}
    }
    batch_docs.append(doc)
    if len(batch_docs) >= 512:
        col.insert_many(batch_docs)
        batch_docs = []
if batch_docs:
    col.insert_many(batch_docs)

print("Inserted documents:", col.count_documents({}))
client.close()
