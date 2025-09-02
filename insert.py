import os, json
from pymongo import MongoClient
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")
SUBCHUNK_FILE = os.getenv("SUBCHUNK_FILE", "subchunk_file.json")  # Update this
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE = os.getenv("DATABASE", "soni_agent")
COLLECTION = os.getenv("COLLECTION", "docs")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

model = SentenceTransformer(MODEL_NAME)

with open(SUBCHUNK_FILE, "r", encoding="utf-8") as f:
    sub_chunks = json.load(f)

client = MongoClient(MONGODB_URI)
col = client[DATABASE][COLLECTION]

inserted = 0
for i in tqdm(range(0, len(sub_chunks), BATCH_SIZE)):
    batch = sub_chunks[i:i+BATCH_SIZE]
    texts = [item["text"] for item in batch]
    vecs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False).tolist()
    docs = [
        {
            "text": item["text"],
            "embedding": v,
            "parent": item["parent"],
            "parent_index": item["parent_index"]
        }
        for item, v in zip(batch, vecs)
    ]
    col.insert_many(docs)
    inserted += len(docs)

print(f"✅ Inserted {inserted} sub-chunks into {DATABASE}.{COLLECTION}")
client.close()