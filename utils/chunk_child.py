import json
import re
from transformers import AutoTokenizer

TOKEN_LIMIT = 50  # Set your desired token count per chunk
MODEL_NAME = "intfloat/multilingual-e5-small"  # Or your preferred model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def split_sentences(text):
    # Simple sentence splitter (can be improved for Vietnamese)
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    return [s.strip() for s in sentences if s.strip()]

def semantic_chunk(sentences, token_limit):
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        if current_tokens + sent_tokens > token_limit and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_tokens = sent_tokens
        else:
            current_chunk.append(sent)
            current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def chunk_json_file(input_path, output_path, token_limit=TOKEN_LIMIT):
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    sub_chunks = []
    for idx, chunk in enumerate(chunks):
        sentences = split_sentences(chunk)
        sem_chunks = semantic_chunk(sentences, token_limit)
        for sub in sem_chunks:
            sub_chunks.append({
                "text": sub,
                "parent": chunk,
                "parent_index": idx
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sub_chunks, f, ensure_ascii=False, indent=2)

chunk_json_file("all_chunk_quychectsv.json", "test1.json")
chunk_json_file("all_chunk_sotaysv.json", "test2.json")
chunk_json_file("all_chunk_qcdt.json", "test3.json")