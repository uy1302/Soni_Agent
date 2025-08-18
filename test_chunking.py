"""
Standalone test script for chunking qcdt.html file
"""
import sys
import os
import json
from datetime import datetime

# Add chunking directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'chunking'))

# Import required modules
from transformers import AutoTokenizer
from llama_index.core.embeddings import MockEmbedding

# Import local modules with absolute imports by modifying them temporarily
import pickle
import base64
import pandas as pd
from io import StringIO
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core import Document
from markdownify import markdownify as md
from bs4 import BeautifulSoup
from typing import Literal, Optional
from pydantic import BaseModel, Field

# Define metadata classes locally to avoid import issues
class ChunkLlamaIndexMetadata(BaseModel):
    data_source_id: str = Field(description="Data source ID")
    source_doc_name: str = Field(description="Source document name")
    source_doc_id: str = Field(description="Source document ID")

class ExpandChunkLlamaIndexMetadata(ChunkLlamaIndexMetadata):
    """Metadata for retrieval purpose in LlamaIndex."""
    type: Literal["text", "table"] = Field(description="Type of content: text or table")
    sub_type: Optional[str] = Field(default=None, description="Sub-type of content (para, titles, etc.)")
    is_table: Optional[bool] = Field(default=False, description="Whether the content is or contains a table")
    table_description: Optional[str] = Field(default=None, description="LLM-generated description of the table")
    table_dataframe: Optional[str] = Field(default=None, description="Base64 encoded serialized dataframe")
    text_here: Optional[str] = Field(default=None, description="The actual text content")

# Define HTMLParser locally
class HTMLParser:
    def __init__(self, table_description_prompt: str, lang: Literal["Vietnamese", "English"], llm_description: LLM = None): 
        self.llm_description = llm_description
        self.lang = lang
        self.excluded_embed_metadata_keys = self.excluded_llm_metadata_keys = list(ExpandChunkLlamaIndexMetadata.model_fields.keys())

    def html_chunking(self, node: BeautifulSoup) -> tuple[list[str], list[str]]:
        if node.name in ["head", "style"]:
            return [], []
        if len(node.findChildren(recursive=False)) == 0:
            return [node.text], [node.name]
        result = []
        tags = []
        for _, child in enumerate(node.findChildren(recursive=False)):
            res = []
            tag = []
            if child.name == 'table':
                res = [child]
                tag = ["table"]
            else:
                res, tag = self.html_chunking(child)
            if isinstance(child.previous_sibling, str) and (child.previous_sibling.strip() != "") and (child.previous_sibling.name != None):
                res = [child.previous_sibling.strip()] + res
                tag = [child.previous_sibling.name] + tag
            if isinstance(child.next_sibling, str) and (child.next_sibling.strip() != "") and (child.next_sibling.name != None):
                res = res + [child.next_sibling.strip()]
                tag = tag + [child.next_sibling.name]
            result.extend(res)
            tags.extend(tag)
        result = [x.strip() if isinstance(x, str) else x for x in result]
        new_result = []
        new_tags = []
        for x, y in zip(result, tags):
            if x != "":
                new_result.append(x)
                new_tags.append(y)
        return new_result, new_tags

    def dict_to_str(self, d: dict) -> str:
        s = base64.b64encode(pickle.dumps(d)).decode('ascii')
        return s

    def soup_table_to_text(self, soup: BeautifulSoup) -> str:
        return md(soup.__str__())

    def replb(self, soup: BeautifulSoup) -> BeautifulSoup:
        for s in soup.select('b'):
            s.extract()
        return soup
    
    def get_title(self, soup: BeautifulSoup) -> str:
        title_tag = soup.find('h2')
        if title_tag:
            return title_tag.text.strip()
        return ""

    def parse(self, html_str: str, base_metadata: ChunkLlamaIndexMetadata) -> tuple[list[Document], str]:
        title = self.get_title(BeautifulSoup(html_str))
        print(f"Title: {title}")
        chunks, tags = self.html_chunking(self.replb(BeautifulSoup(html_str)))
        docs = []
        texts = []
        
        for chunk, tag in zip(chunks, tags):
            if tag != "table":
                chunk = chunk.strip()
                if tag.startswith("h"):
                    if len(texts) > 0:
                        doc_metadata = ExpandChunkLlamaIndexMetadata(
                            **base_metadata.model_dump(),
                            type="text",
                            text_here="\n".join(texts),
                            sub_type="para"
                        )
                        docs.append(Document(
                            text="\n".join(texts), 
                            metadata=doc_metadata.model_dump(),
                            excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                            excluded_llm_metadata_keys=self.excluded_llm_metadata_keys
                        ))
                    texts = [chunk]
                else:
                    texts.append(chunk)
            else:
                if len(texts) > 0:
                    doc_metadata = ExpandChunkLlamaIndexMetadata(
                        **base_metadata.model_dump(),
                        type="text",
                        text_here="\n".join(texts),
                        sub_type="para"
                    )
                    docs.append(Document(
                        text="\n".join(texts), 
                        metadata=doc_metadata.model_dump(),
                        excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=self.excluded_llm_metadata_keys
                    ))

                texts = []
                table_text = self.soup_table_to_text(chunk)

                try:
                    table_dataframe = self.dict_to_str(pd.read_html(StringIO(chunk.__str__()))[0].to_dict())
                except Exception as e:
                    print(f"Error converting table to dataframe: {e}")
                    table_dataframe = None

                table_text_metadata = ExpandChunkLlamaIndexMetadata(
                    **base_metadata.model_dump(),
                    type="text",
                    text_here=table_text,
                    sub_type="para",
                    is_table=True,
                    table_dataframe=table_dataframe
                )
                docs.append(Document(
                    text=table_text, 
                    metadata=table_text_metadata.model_dump(),
                    excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=self.excluded_llm_metadata_keys
                ))

        if len(texts) > 0:
            doc_metadata = ExpandChunkLlamaIndexMetadata(
                **base_metadata.model_dump(),
                type="text",
                text_here="\n".join(texts),
                sub_type="para"
            )
            docs.append(Document(
                text="\n".join(texts), 
                metadata=doc_metadata.model_dump(),
                excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=self.excluded_llm_metadata_keys
            ))

        if title:
            for doc in docs:
                doc.metadata["title"] = title

        titles = []
        for chunk, tag in zip(chunks, tags):
            if isinstance(chunk, str) and tag.startswith("h"):
                titles.append(chunk)

        if len(titles) > 0:
            titles_text = "\n".join(titles)
            titles_metadata = ExpandChunkLlamaIndexMetadata(
                **base_metadata.model_dump(),
                type="text",
                text_here=titles_text,
                sub_type="titles"
            )
            docs.append(Document(
                text=titles_text, 
                metadata=titles_metadata.model_dump(),
                excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=self.excluded_llm_metadata_keys
            ))

        return docs, title

# Define Splitter locally
class Splitter: 
    def __init__(self, tokenizer: AutoTokenizer, seperator: str="\n"):
        self.tokenizer = tokenizer
        self.seperator = seperator
        self.max_num_tables = 1
        self.excluded_embed_metadata_keys = self.excluded_llm_metadata_keys = list(ExpandChunkLlamaIndexMetadata.model_fields.keys())
        # Use a reasonable default max length to avoid issues
        self.model_max_length = 500  # Set a reasonable default

    def count_tokens(self, text: str) -> int:
        """Simple token counting with error handling"""
        try:
            if len(text) > 10000:  # If text is too long, estimate tokens
                return len(text) // 4  # Rough estimate: 4 chars per token
            return len(self.tokenizer.tokenize(text))
        except:
            return len(text) // 4  # Fallback estimation

    def merge_docs(self, docs: list[Document], title: str) -> Document:
        new_text = f"Tiêu đề: {title}\n{self.seperator.join([doc.text for doc in docs])}"
        metadata = docs[0].metadata.copy()
        metadata["sub_type"] = "merge"
        metadata["text_here"] = new_text
        new_doc = Document(
            text=new_text,
            metadata=metadata,
            excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
        )
        return new_doc

    def shorten(self, s: str) -> str:
        """Simple text shortening based on estimated tokens"""
        if len(s) <= self.model_max_length * 4:  # Rough estimation
            return s
        return s[:self.model_max_length * 4]

    def process_long_text(self, doc: Document) -> list[Document]:
        """Simplified version to avoid tokenizer issues"""
        # If text is reasonably short, return as-is
        if len(doc.text) <= self.model_max_length * 4:
            return [doc]
        
        # Split into smaller chunks
        texts = doc.text.split(self.seperator)
        new_docs = []
        current_text = ""
        
        for text in texts:
            test_text = current_text + self.seperator + text if current_text else text
            if len(test_text) <= self.model_max_length * 4:
                current_text = test_text
            else:
                if current_text:
                    new_docs.append(Document(
                        text=current_text,
                        metadata=doc.metadata.copy(),
                        excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
                    ))
                current_text = text
        
        if current_text:
            new_docs.append(Document(
                text=current_text,
                metadata=doc.metadata.copy(),
                excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
            ))
        
        return new_docs if new_docs else [doc]

    def dedup(self, docs: list[Document]) -> list[Document]:
        s = set()
        new_docs = []
        for doc in docs:
            dict_doc = {
                "text": doc.text,
                "metadata": doc.metadata,
                "excluded_embed_metadata_keys": self.excluded_embed_metadata_keys,
                "excluded_llm_metadata_keys": self.excluded_llm_metadata_keys,
            }
            str_doc = json.dumps(dict_doc, ensure_ascii=False)
            if str_doc not in s:
                new_docs.append(doc)
                s.add(str_doc)
        return new_docs

    def __call__(self, docs: list[Document], title: str) -> list[Document]:
        """Simplified chunking process"""
        new_docs = []
        
        # Process each document individually for simplicity
        for doc in docs:
            processed_docs = self.process_long_text(doc)
            new_docs.extend(processed_docs)
        
        # Simple deduplication
        new_docs = self.dedup(new_docs)
        return new_docs

def main():
    # Initialize components
    table_description_prompt = ""
    lang = "Vietnamese"
    
    parser = HTMLParser(
        llm_description=None,
        table_description_prompt=table_description_prompt,
        lang=lang
    )
    
    # Initialize splitter
    pretrained_model_name = "AITeamVN/Vietnamese_Embedding"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    splitter = Splitter(
        tokenizer=tokenizer,
        seperator="\n"
    )
    
    # Read the HTML file
    html_str = open("data/qcdt.html", "r", encoding="utf-8").read()
    
    # Create metadata for the document
    metadata = ChunkLlamaIndexMetadata(
        data_source_id="cmc_1",
        source_doc_id="qcdt.html-0",
        source_doc_name="qcdt.html",
    )
    
    # Test chunking process
    print("Starting chunking process...")
    docs, title = parser.parse(html_str, metadata)
    print(f"Title extracted: {title}")
    print(f"Number of documents from parser: {len(docs)}")
    
    # Test splitting process
    print("\nStarting splitting process...")
    chunks = splitter(docs, title=title)
    print(f"Number of chunks after splitting: {len(chunks)}")
    
    # Prepare data for saving
    chunks_data = {
        "title": title,
        "processing_info": {
            "timestamp": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "original_docs": len(docs),
            "model_max_length": splitter.model_max_length
        },
        "chunks": []
    }
    
    # Collect token statistics (using simple estimation)
    token_counts = [len(chunk.text) // 4 for chunk in chunks]  # Rough estimation
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        chunk_info = {
            "chunk_id": i + 1,
            "text": chunk.text,
            "text_length": len(chunk.text),
            "token_count": token_counts[i],
            "metadata": chunk.metadata,
            "doc_id": chunk.doc_id if hasattr(chunk, 'doc_id') else None,
        }
        chunks_data["chunks"].append(chunk_info)
    
    # Add statistics
    chunks_data["statistics"] = {
        "average_tokens_per_chunk": sum(token_counts) / len(token_counts),
        "max_tokens": max(token_counts),
        "min_tokens": min(token_counts),
        "chunks_exceeding_limit": sum(1 for count in token_counts if count > splitter.model_max_length),
        "token_distribution": {
            "0-100": sum(1 for count in token_counts if 0 <= count <= 100),
            "101-200": sum(1 for count in token_counts if 101 <= count <= 200),
            "201-300": sum(1 for count in token_counts if 201 <= count <= 300),
            "301-400": sum(1 for count in token_counts if 301 <= count <= 400),
            "400+": sum(1 for count in token_counts if count > 400)
        }
    }
    
    # Save to JSON file
    output_file = "data/qcdt_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nChunks saved to: {output_file}")
    
    # Display summary information
    print("\n=== Chunk Analysis Summary ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average tokens per chunk: {chunks_data['statistics']['average_tokens_per_chunk']:.2f}")
    print(f"Max tokens in a chunk: {chunks_data['statistics']['max_tokens']}")
    print(f"Min tokens in a chunk: {chunks_data['statistics']['min_tokens']}")
    print(f"Model max length: {splitter.model_max_length}")
    
    # Token distribution
    print("\n=== Token Distribution ===")
    for range_name, count in chunks_data['statistics']['token_distribution'].items():
        print(f"  {range_name} tokens: {count} chunks")
    
    # Check for chunks that might be too long
    if chunks_data['statistics']['chunks_exceeding_limit'] > 0:
        print(f"\nWarning: {chunks_data['statistics']['chunks_exceeding_limit']} chunks exceed model max length!")
        long_chunks = [(i, count) for i, count in enumerate(token_counts) if count > splitter.model_max_length]
        for idx, token_count in long_chunks[:3]:  # Show first 3 long chunks
            print(f"  Chunk {idx + 1}: {token_count} tokens")
    else:
        print("\nAll chunks are within token limits ✓")
    
    # Show chunk types
    chunk_types = {}
    for chunk in chunks:
        chunk_type = chunk.metadata.get('type', 'unknown')
        sub_type = chunk.metadata.get('sub_type', 'none')
        key = f"{chunk_type}:{sub_type}"
        chunk_types[key] = chunk_types.get(key, 0) + 1
    
    print("\n=== Chunk Types ===")
    for chunk_type, count in chunk_types.items():
        print(f"  {chunk_type}: {count} chunks")
    
    print(f"\nDetailed results saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()
