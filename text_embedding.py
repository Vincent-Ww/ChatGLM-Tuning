# -- coding: utf-8 --
# @Author : wuzixun
# @Time : 2023/5/22 5:07 PM
# @File : text_embedding

from __future__ import annotations

from typing import List

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings as ChromaDBEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


class TextEmbedding(Embeddings):
    """
    customized text embedding class inherit from langchain embeddings interface
    """

    def __init__(self):
        self.sentence_embedding = self.load_sentence_embedding()
 
    @staticmethod
    def load_sentence_embedding() -> SentenceTransformer | None:
        return SentenceTransformer("GanymedeNil/text2vec-large-chinese").cuda()

    def _get_sentence_embedding(self, document):
        return self.sentence_embedding.encode(document, device="cuda:0").tolist()

    def embed(self, document) -> List[float]:
        """
        Embeds the given document into a vector.

        :param document: The document to embed.
        :return: The embedding vector.
        """

        if self.sentence_embedding is not None:
            return self._get_sentence_embedding(document)
        raise NotImplementedError("TextEmbedding has not been initialized.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed(text)


class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> ChromaDBEmbeddings:
        # embed the documents somehow
        return []
