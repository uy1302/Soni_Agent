import os
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from utils import *

load_dotenv()

# --- DB Config ---
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE    = os.getenv("DATABASE", "soni_agent")
COLLECTION  = os.getenv("COLLECTION", "docs")
INDEX_NAME  = os.getenv("INDEX_NAME", "default")
TOP_K       = int(os.getenv("TOP_K", "3"))
NUM_CAND    = int(os.getenv("NUM_CANDIDATES", "100"))
E5_MODEL    = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.2"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

PREVIEW_CHARS = 3000

mongo = MongoClient(MONGODB_URI)
col = mongo[DATABASE][COLLECTION]
embedder = SentenceTransformer(E5_MODEL)

@tool("DatabaseSearch", return_direct=False)
def db_search_tool(query: str) -> str:
    """Tìm kiếm thông tin trong cơ sở dữ liệu nội bộ bằng RAG."""
    qvec = embedder.encode([query], show_progress_bar=False)[0].tolist()
    cursor = col.aggregate([
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": qvec,
                "numCandidates": NUM_CAND,
                "limit": TOP_K   
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
    results: List[Dict] = [doc for doc in cursor if doc["score"] >= SCORE_THRESHOLD]
    if not results:
        return "Không tìm thấy dữ liệu trong DB."
    context_lines = []
    for i, doc in enumerate(results, 1):
        src = f"(parent_index={doc.get('parent_index', '')})" if "parent_index" in doc else ""
        context_lines.append(f"[{i}] {src}\n{doc['parent']}")
    return "\n\n".join(context_lines)

@tool("InternetSearch", return_direct=False)
def internet_search_tool(query: str) -> str:
    """Tìm kiếm thông tin qua Internet khi không tìm thấy kết quả trong cơ sở dữ liệu nội bộ."""
    q = (query or "").strip()
    if "đại học bách khoa hà nội" not in q.lower():
        q = f"{q} Đại học Bách Khoa Hà Nội"
    try:
        resp = tavily_client.search(q, max_results=5)
    except Exception as e:
        return f"[Lỗi khi gọi Tavily: {e}]"
    raw_results = resp.get("results") if isinstance(resp, dict) else resp
    if not raw_results:
        return "[Không tìm thấy thông tin trên Internet]"
    contexts = []
    for idx, item in enumerate(raw_results, start=1):
        url = None
        title = ""
        score = None
        snippet = ""
        if isinstance(item, dict):
            url = item.get("url") or item.get("link") or item.get("href")
            title = item.get("title") or item.get("name") or ""
            score = item.get("score")
            snippet = item.get("snippet") or item.get("description") or ""
        elif isinstance(item, tuple) and len(item) >= 1:
            url = item[0]
            if len(item) > 1:
                score = item[1]
        elif isinstance(item, str):
            url = item
        if not url:
            continue
        page_text = fetch_page_content(url)
        if not isinstance(page_text, str):
            page_text = str(page_text)
        preview = page_text.strip().replace("\n", " ")
        if len(preview) > PREVIEW_CHARS:
            preview = preview[:PREVIEW_CHARS] + " ...[cut]"
        meta = f"URL: {url}"
        if score is not None:
            meta += f" | score: {score}"
        header = f"[{idx}] {title}".strip()
        if header == f"[{idx}]":
            header = f"[{idx}] (No title)"
        contexts.append(f"{header}\n{meta}\n{preview}")
    return "\n\n".join(contexts) if contexts else "[Không tìm thấy thông tin trên Internet]"