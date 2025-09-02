import os, textwrap
from typing import List, Dict
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Gemini SDK
from google import genai
from google.genai import types

load_dotenv()

# ---- Config ----
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE    = os.getenv("DATABASE", "soni_agent")
COLLECTION  = "new_docs"
INDEX_NAME  = os.getenv("INDEX_NAME", "default")

TOP_K       = int(os.getenv("TOP_K", "3"))
NUM_CAND    = int(os.getenv("NUM_CANDIDATES", "100"))

E5_MODEL    = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")
GEMINI_FLASH = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.0-flash-001")

mongo = MongoClient(MONGODB_URI)
col = mongo[DATABASE][COLLECTION]

embedder = SentenceTransformer(E5_MODEL)
gem_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    qvec = embedder.encode([query], show_progress_bar=False)[0].tolist()
    cursor = col.aggregate([
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": qvec,
                "numCandidates": NUM_CAND,
                "limit": top_k   
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
                "parent": 1,
                "parent_index": 1
            }
        }
    ])
    # Score threshold for relevance
    SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
    results = [doc for doc in cursor if doc["score"] >= SCORE_THRESHOLD]
    # For each result, return parent chunk for context
    final = []
    for doc in results[:top_k]:
        final.append({
            "text": doc["parent"],
            "meta": {"index": doc["parent_index"]},
            "score": doc["score"]
        })
    return final

SYSTEM_HINT = """You are a precise RAG assistant. Answer in Vietnamese.
- Chỉ dựa vào 'Context' dưới đây, không bịa thêm.
- Nếu có, hãy trích dẫn Điều/Khoản rõ ràng.
- Trả lời ngắn gọn, rõ ràng, có gạch đầu dòng khi hợp lý.
- Cuối cùng liệt kê nguồn [#] theo Context.
"""

def build_context(chunks: List[Dict]) -> str:
    lines = []
    for i, ch in enumerate(chunks, 1):
        meta = ch.get("meta", {})
        src = f"(index={meta.get('index')})" if "index" in meta else ""
        lines.append(f"[{i}] {src}\n{ch['text']}")
    return "\n\n".join(lines)

def generate_answer(query: str, contexts: List[Dict]) -> str:
    context_block = build_context(contexts)
    user_prompt = f"""{SYSTEM_HINT}

Câu hỏi: {query}

Context:
{context_block}
"""
    resp = gem_client.models.generate_content(
        model=GEMINI_FLASH,
        contents=[
            {"role": "user", "parts": [{"text": user_prompt}]},
        ],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=800
        ),
    )
    return resp.text


if __name__ == "__main__":
    while True:
        q = input("\nNhập câu hỏi (hoặc 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        hits = retrieve(q, TOP_K)
        if not hits:
            print("Không tìm thấy ngữ cảnh phù hợp.")
            continue

        print("\nTop hits:")
        for i, h in enumerate(hits, 1):
            preview = h["text"].replace("\n", " ")
            print(f"[{i}] score={h['score']:.4f}  {preview}...")

        ans = generate_answer(q, hits)
        print("\n--- Trả lời ---")
        print(textwrap.fill(ans, width=100))
