import json
from llama_index.core import Document
# from .tokenizer import CustomTokenizer
from metadata import ExpandChunkLlamaIndexMetadata
from transformers import AutoTokenizer

class Splitter: 
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        seperator: str="\n",
    ):
        """
        Args:
            tokenizer (CustomTokenizer): The tokenizer to use for splitting text.
            seperator (str): The separator to use for splitting text. Default is "\n".
        """
        self.tokenizer = tokenizer
        self.seperator = seperator
        self.max_num_tables = 1
        self.excluded_embed_metadata_keys = self.excluded_llm_metadata_keys = list(ExpandChunkLlamaIndexMetadata.model_fields.keys())
        try:
            self.model_max_length = self.tokenizer.model_max_length - 10
        except:
            self.model_max_length = 512

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

    def merge_docs_title(self, docs: list[Document]) -> Document:
        metadata_copy = docs[0].metadata.copy()
        new_text = "\n".join([doc.text for doc in docs])
        metadata_copy["text_here"] = new_text
        return Document(
            text=new_text,
            metadata=metadata_copy,
            excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
        )

    def shorten(self, s: str) -> str:
        return self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(s)[:self.model_max_length]
            )
        )

    def process_long_text(self, doc: Document) -> list[Document]:
        texts = doc.text.split(self.seperator)
        i = j = 0
        s = ""
        new_docs = []
        current_i = None
        while True:
            if i >= len(texts):
                break
            if len(self.tokenizer.tokenize(texts[i])) > self.model_max_length:
                if s != "":
                    new_docs.append(Document(
                        text=self.shorten(s),
                        metadata=doc.metadata.copy(),
                        excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
                    ))

                new_docs.append(Document(
                    text=self.shorten(texts[i]),
                    metadata=doc.metadata.copy(),
                    excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
                ))
                i += 1
                j = i
                current_i = i
            elif len(self.tokenizer.tokenize(self.seperator.join([s, texts[i]]).strip())) <= self.model_max_length:
                s = self.seperator.join([s, texts[i]])
                if i == len(texts) - 1:
                    new_docs.append(Document(
                        text=self.shorten(s),
                        metadata=doc.metadata.copy(),
                        excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
                    ))
                i += 1
            else:
                new_doc = Document(
                    text=self.shorten(s),
                    metadata=doc.metadata.copy(),
                    excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
                )
                new_docs.append(new_doc)
                if j + 1 < i:
                    j += 1
                    if current_i != i:
                        new_docs.append(Document(
                            text=self.shorten(s),
                            metadata=doc.metadata.copy(),
                            excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                            excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
                        ))
                        current_i = i
                    s = self.seperator.join(texts[j: i])
                else:
                    s = ""
                    i += 1
                    j = i
        return new_docs

    def shorten_docs(self, docs: list[Document]) -> list[Document]:
        new_docs = []
        for doc in docs:
            new_docs.append(Document(
                text=self.shorten(doc.text),
                metadata=doc.metadata.copy(),
                excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
            ))
        return new_docs

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
        new_docs = docs
        docs_normal = [
            doc for doc in docs if ("sub_type" in doc.metadata) and (doc.metadata["sub_type"] == "para")
        ]

        set_indices = set()
        for i, doc in enumerate(docs_normal):
            j = 1
            consider_docs = [doc]
            consider_text = doc.text
            indices = f"{i}"
            expand_up = True
            expand_down = True
            num_table = int(doc.metadata.get("is_table", "") == "true")
            while True:
                if expand_up and (i - j >= 0) and (num_table + int(docs_normal[i - j].metadata.get("is_table", "") == "true") <= self.max_num_tables):
                    tmp_text = docs_normal[i - j].text.strip() + self.seperator + consider_text
                    if len(self.tokenizer.tokenize(tmp_text)) <= self.model_max_length:
                        consider_docs = [docs_normal[i - j]] + consider_docs
                        consider_text = tmp_text
                        expand_up = True
                        indices = f"{i - j}::{indices}"
                        num_table = num_table + int(docs_normal[i - j].metadata.get("is_table", "") == "true")
                    else:
                        expand_up = False
                else:
                    expand_up = False

                if expand_down and (i + j < len(docs_normal)) and (num_table + int(docs_normal[i + j].metadata.get("is_table", "") == "true") <= self.max_num_tables):
                    tmp_text = consider_text + self.seperator + docs_normal[i + j].text.strip()
                    if len(self.tokenizer.tokenize(tmp_text)) <= self.model_max_length:
                        consider_docs = consider_docs + [docs_normal[i + j]]
                        consider_text = tmp_text
                        expand_down = True
                        indices = f"{indices}::{i + j}"
                        num_table = num_table + int(docs_normal[i + j].metadata.get("is_table", "") == "true")
                    else:
                        expand_down = False
                else:
                    expand_down = False

                if (expand_down == False) and (expand_up == False):
                    break
                j += 1

            if indices not in set_indices:
                set_indices.add(indices)
                new_docs.append(self.merge_docs(
                       consider_docs,
                       title=title
                    )
                )

        docs_title = [doc for doc in docs if ("sub_type" in doc.metadata) and (doc.metadata["sub_type"] == "titles")]
        if len(docs_title) > 0:
            new_doc_title = self.merge_docs_title(docs_title)
            new_docs = [doc for doc in new_docs if not (("sub_type" in doc.metadata) and (doc.metadata["sub_type"] == "titles"))]
            new_docs.append(new_doc_title)

        new_new_docs = []
        for doc in new_docs:
            new_new_docs.extend(self.process_long_text(doc))

        new_new_docs = self.dedup(new_new_docs)
        return new_new_docs