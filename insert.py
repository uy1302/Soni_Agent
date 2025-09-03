import os, json
from pymongo import MongoClient
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE = os.getenv("DATABASE", "soni_agent")
COLLECTION = os.getenv("COLLECTION", "docs")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

model = SentenceTransformer(MODEL_NAME)

SUBCHUNK_FILES = ["test1.json", "test2.json", "test3.json"]

sub_chunks = []
for subchunk_file in SUBCHUNK_FILES:
    with open(subchunk_file, "r", encoding="utf-8") as f:
        sub_chunks.extend(json.load(f))

client = MongoClient(MONGODB_URI)
col = client[DATABASE][COLLECTION]

inserted = 0
for i in tqdm(range(0, len(sub_chunks), BATCH_SIZE)):
    batch = sub_chunks[i:i+BATCH_SIZE]
    items = []
    for item in batch:
        items.append(item)
    texts = [item["text"] for item in items]
    vecs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False).tolist()
    docs = [
        {
            "text": item["text"],
            "embedding": v,
            "parent": item["parent"],
            "parent_index": item["parent_index"]
        }
        for item, v in zip(items, vecs)
    ]
    col.insert_many(docs)
    inserted += len(docs)

print(f"✅ Inserted {inserted} sub-chunks into {DATABASE}.{COLLECTION}")
client.close()