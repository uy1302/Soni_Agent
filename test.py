import os, json
from pymongo import MongoClient
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")
CHUNKS_FILE = os.getenv("CHUNKS_FILE", "all_chunk_qcdt.json")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE = os.getenv("DATABASE", "soni_agent")
COLLECTION = os.getenv("COLLECTION", "docs")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

model = SentenceTransformer(MODEL_NAME)

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    texts = json.load(f)

client = MongoClient(MONGODB_URI)
col = client[DATABASE][COLLECTION]

inserted = 0
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i:i+BATCH_SIZE]
    vecs = model.encode(batch, batch_size=BATCH_SIZE, show_progress_bar=False).tolist()
    docs = [
        {"text": t, "embedding": v, "meta": {"index": i+j}}
        for j, (t,v) in enumerate(zip(batch, vecs))
    ]
    col.insert_many(docs)
    inserted += len(docs)

print(f"✅ Inserted {inserted} docs into {DATABASE}.{COLLECTION}")
client.close()
