import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
import requests
from tavily import TavilyClient
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient


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

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def fetch_page_content(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        texts = soup.stripped_strings
        return "\n".join(texts)
    except Exception as e:
        return f"[Lỗi khi truy cập {url}: {e}]"

@tool("InternetSearch", return_direct=False)
def internet_search_tool(query: str) -> str:
    """Tìm kiếm thông tin trên Internet khi dữ liệu nội bộ không đủ."""
    results = tavily_client.search(query + "Đại học Bách Khoa Hà Nội", max_results=3)
    contexts = []
    for i, res in enumerate(results.get("results", []), 1):
        url = res.get("url")
        title = res.get("title", "")
        page_text = fetch_page_content(url)
        contexts.append(f"[{i}] {title}\nURL: {url}\n{page_text[:2000]}")
    return "\n\n".join(contexts) if contexts else "[Không tìm thấy thông tin trên Internet]"
