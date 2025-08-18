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

# ---- Search function ----
def search(query: str, top_k: int = 5):
    # Encode query
    query_vec = model.encode([query])[0].tolist()
    
    # Run vector search
    results = col.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_vec,
                "path": "embedding",
                "numCandidates": 100,   
                "limit": top_k,         
                "index": "default"      
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

# ---- Example ----
if __name__ == "__main__":
    query = "Điều kiện bảo vệ luận án tiến sĩ"
    hits = search(query, top_k=5)
    for h in hits:
        print(f"Score: {h['score']:.4f} | Text: {h['text'][:100]}...")
