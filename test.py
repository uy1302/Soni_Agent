import json
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

with open("all_chunk_qcdt.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

documents = [Document(text=chunk) for chunk in chunks]


embed_model = HuggingFaceEmbedding(model_name="AITeamVN/Vietnamese_Embedding")

vector_store = PGVectorStore.from_params(
    database="postgres",
    host="localhost",
    password="123456",
    port="5432",
    user="postgres",
    table_name="qcdt_test",
    embed_dim=768,   
    hybrid_search=True,
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

pipeline = IngestionPipeline(
    transformations=[embed_model],
    vector_store=vector_store,
)

pipeline.run(documents)
print("✅ Data inserted into Postgres successfully")
