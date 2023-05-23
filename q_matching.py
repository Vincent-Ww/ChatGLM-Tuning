# -- coding: utf-8 --
# @Author : wuzixun
# @Time : 2023/5/22 4:54 PM
# @File : q_matching

import json
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from text_embedding import TextEmbedding


class Retrieval(object):
    def __init__(self, load_from_disk=False, persist_directory=None):
        self.embedding = TextEmbedding()
        self.persist_directory = persist_directory
        self.__init_vectordb(load_from_disk)

    def __init_vectordb(self, load_from_disk):
        if load_from_disk:
            self.db = Chroma(collection_name='h2h-questions', persist_directory=self.persist_directory, embedding_function=self.embedding)
            return 
        with open("data/h2h_question/q2ft.json", "r") as f:
            q2ft = json.load(f)
        h2h_q_names = q2ft.keys()
        questions = [Document(page_content=q) for q in h2h_q_names]
        self.db = Chroma.from_documents(questions, embedding=self.embedding, collection_name='h2h-questions', persist_directory=self.persist_directory)
        self.db.persist()
        print("persist vector database to disk")

    def retrieve(self, query):
        retrived_res = self.db.as_retriever(search_type="similarity", search_kwargs={'k': 1}).get_relevant_documents(query)
        retrieved_q = retrived_res[0].page_content
        return retrieved_q


#retrieval = Retrieval(load_from_disk=True, persist_directory=".chroma/biaozhunwen")
#print(retrieval.retrieve("远洋夺宝是什么怎么玩的啊"))
