import os
from pymongo import MongoClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load env
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE = os.getenv("DATABASE", "soni_agent")
COLLECTION = os.getenv("COLLECTION", "docs")

# Load model + db
model = SentenceTransformer(MODEL_NAME)
client = MongoClient(MONGODB_URI)
col = client[DATABASE][COLLECTION]

def search(query: str, top_k: int = 5):
    query_vec = model.encode([query])[0].tolist()
    results = col.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_vec,
                "path": "embedding",
                "numCandidates": 100,
                "limit": top_k,
                "index": "default"  # Atlas Search index name
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ])
    return list(results)

if __name__ == "__main__":
    while True:
        q = input("\nEnter your query (or 'exit' to quit): ")
        if q.lower() in ["exit", "quit"]:
            break
        hits = search(q, top_k=5)
        if not hits:
            print("No results found.")
        for h in hits:
            print(f"Score: {h['score']:.4f} | Text: {h['text'][:100]}...")
