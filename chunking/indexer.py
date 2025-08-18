from typing import Literal
from loguru import logger
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import Document
from llama_index.core.settings import Settings
from llama_index.core.storage import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding

from parser import HTMLParser
from splitter import Splitter
from metadata import ChunkLlamaIndexMetadata


class Indexer:
    def __init__(
        self,
        parser: HTMLParser,
        splitter: Splitter,
        embed_model: BaseEmbedding,
        vector_store: PGVectorStore,
    ):
        self.parser = parser
        self.splitter = splitter
        self.vector_store = vector_store
        Settings.embed_model = embed_model
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)

    @staticmethod
    def filter_docs_by_type(docs: list[Document], doc_type: Literal["text", "table"]) -> list[Document]:
        return [doc for doc in docs if doc.metadata["type"] == doc_type]
    
    def index(self, chunk: Document):
        try:
            _ = VectorStoreIndex.from_documents(
                documents=[chunk],
                storage_context=self.storage_context,
                show_progress=True
            )
            logger.info(f"Indexed {chunk.doc_id} successfully")
 
        except Exception as e:
            logger.error(f"Error indexing document: {chunk.doc_id}")
            raise e
 
        
    def chunk(self, html: str, metadata: ChunkLlamaIndexMetadata) -> list[Document]:
        docs, title = self.parser.parse(html, metadata)
        text_docs = self.filter_docs_by_type(docs, "text")
        table_docs = self.filter_docs_by_type(docs, "table")
        text_chunks = self.splitter(text_docs, title=title)
        table_chunks = self.splitter.shorten_docs(table_docs)
        return table_chunks + text_chunks
 
