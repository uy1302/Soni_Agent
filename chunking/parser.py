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

from metadata import ExpandChunkLlamaIndexMetadata, ChunkLlamaIndexMetadata


class HTMLParser:
    def __init__(self, table_description_prompt: str, lang: Literal["Vietnamese", "English"], llm_description: LLM = None,): 
        # TODO: change llm_description_prompt from str to langfuse intergration
        # self.llm_description = llm_description
        # self.table_description_template = PromptTemplate(
        #     template=table_description_prompt
        # )
        self.llm_description = llm_description
        # self.table_description_template = PromptTemplate(
        #     template=table_description_prompt
        # ) 
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

    def process_output_llm_description(self, text: str) -> str:
        pos = text.find(f"Summarization in {self.lang}:")
        if pos > -1:
            return text[pos + len(f"Summarization in {self.lang}:"): ].strip()
        return text
    
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
        # table_description_index = 0
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
                table_text = self.soup_table_to_text(chunk)  # chunk: HTML, table_text: markdown

                # table_description = self.process_output_llm_description(
                #     self.llm_description.complete(
                #         prompt=self.table_description_template.format(
                #             lang=self.lang,
                #             context=table_text
                #         )
                #     ).text
                # ) # for retrieval, choose table
 
                try:
                    table_dataframe = self.dict_to_str(pd.read_html(StringIO(chunk.__str__()))[0].to_dict())
                except Exception as e:
                    print(f"Error converting table to dataframe: {e}")
                    table_dataframe = None

                # # Table document with description as text
                # table_metadata = ExpandChunkLlamaIndexMetadata(
                #     **base_metadata.model_dump(),
                #     type="table",
                #     is_table=True,
                #     table_description=table_description,
                #     table_dataframe=table_dataframe
                # )
                # docs.append(Document(
                #     text=table_description, 
                #     metadata=table_metadata.model_dump(),
                #     excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                #     excluded_llm_metadata_keys=self.excluded_llm_metadata_keys
                # ))

                # Table markdown text as a separate document
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

        # Add any remaining text chunks
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

        # Append title to all documents's metadata if available, this metadata will be used for retrieval
        if title:
            for doc in docs:
                doc.metadata["title"] = title

        # Extract titles for a special document
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