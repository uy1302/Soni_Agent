import os, textwrap, requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
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
COLLECTION  = os.getenv("COLLECTION", "docs")
INDEX_NAME  = os.getenv("INDEX_NAME", "default")

TOP_K       = int(os.getenv("TOP_K", "1"))
NUM_CAND    = int(os.getenv("NUM_CANDIDATES", "100"))

E5_MODEL    = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")
GEMINI_FLASH = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-pro")

mongo = MongoClient(MONGODB_URI)
col = mongo[DATABASE][COLLECTION]

embedder = SentenceTransformer(E5_MODEL)
gem_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

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

    SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.2"))
    results = [doc for doc in cursor if doc["score"] >= SCORE_THRESHOLD]
    final = []
    for doc in results[:top_k]:
        final.append({
            "text": doc["parent"],
            "meta": {"index": doc["parent_index"]},
            "score": doc["score"],
            "search": doc["text"]
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
            max_output_tokens=2000
        ),
    )
    return resp.text


def fetch_page_content(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Get all visible text
        texts = soup.stripped_strings
        return "\n".join(texts)
    except Exception as e:
        return f"[Lỗi khi truy cập {url}: {e}]"

def tavily_internet_search(query: str, max_links: int = 3) -> str:
    # Search with Tavily
    results = tavily_client.search(query, max_results=max_links)
    contexts = []
    for i, res in enumerate(results.get("results", []), 1):
        url = res.get("url")
        title = res.get("title", "")
        page_text = fetch_page_content(url)
        contexts.append(f"[{i}] {title}\nURL: {url}\n{page_text[:2000]}")  # Limit to 2000 chars per page
    return "\n\n".join(contexts) if contexts else "[Không tìm thấy thông tin trên Internet]"

def agent_answer(query: str) -> str:
    hits = retrieve(query, TOP_K)
    if not hits:
        print("Không tìm thấy ngữ cảnh phù hợp trong dữ liệu. Đang tìm trên Internet...")
        # Use Tavily to get context from web pages
        internet_context = tavily_internet_search(query)
        user_prompt = f"""Bạn là trợ lý AI. Trả lời ngắn gọn, rõ ràng bằng tiếng Việt, dựa trên thông tin sau:
{internet_context}
Câu hỏi: {query}
(Nếu không tìm thấy thông tin, hãy nói rõ là không có dữ liệu trên Internet.)"""
        resp = gem_client.models.generate_content(
            model=GEMINI_FLASH,
            contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
            config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=2000
            ),
        )
        return f"[Thông tin từ Internet]\n{resp.text}"
    ans = generate_answer(query, hits)
    if not ans.strip() or "không biết" in ans.lower() or "không có thông tin" in ans.lower():
        print("Không có câu trả lời rõ ràng từ dữ liệu. Đang tìm trên Internet...")
        internet_context = tavily_internet_search(query)
        user_prompt = f"""Bạn là trợ lý AI. Trả lời ngắn gọn, rõ ràng bằng tiếng Việt, dựa trên thông tin sau:
{internet_context}
Câu hỏi: {query}
(Nếu không tìm thấy thông tin, hãy nói rõ là không có dữ liệu trên Internet.)"""
        resp = gem_client.models.generate_content(
            model=GEMINI_FLASH,
            contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
            config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=2000
            ),
        )
        return f"[Thông tin từ Internet]\n{resp.text}"
    return ans


if __name__ == "__main__":
    while True:
        q = input("\nNhập câu hỏi (hoặc 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        ans = agent_answer(q)
        print("\n--- Trả lời ---")
        print(textwrap.fill(ans, width=100))
