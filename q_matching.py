# -- coding: utf-8 --
# @Author : wuzixun
# @Time : 2023/5/22 4:54 PM
# @File : q_matching

import pandas as pd
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from text_embedding import TextEmbedding


class Retrieval(object):
    def __init__(self, load_from_disk=False, persist_directory=None):
        self.embedding = TextEmbedding()
        self.__init_vectordb(load_from_disk)
        self.persist_directory = persist_directory

    def __init_vectordb(self, load_from_disk):
        if load_from_disk:
            self.db = Chroma(collection_name='h2h-questions', persist_directory=self.persist_directory, embedding_function=self.embedding)
        part1 = pd.read_csv("data/h2h_question/part1.csv")
        part2 = pd.read_csv("data/h2h_question/part2.csv")
        combine = pd.concat([part1, part2], axis=0)
        h2h_q_names = combine['title'].to_list()
        h2h_q_ids = combine['qid'].to_list()
        questions = [Document(page_content=q, metadata={"qid": qid}) for q, qid in zip(h2h_q_names, h2h_q_ids)]
        self.db = Chroma().from_documents(questions, embedding=self.embedding, persist_directory=self.persist_directory)
        if self.persist_directory:
            self.db.persist()
            print("persist vector database to disk")
        print("init vector database: done")

    def retrieve(self, query):
        retrived_res = self.db.as_retriever(search_type="similarity", search_kwargs={'k': 1}).get_relevant_documents(query)
        retrieved_q = retrived_res[0].page_content
        retrieved_qid = retrived_res[0].metadata['qid']
        return {"retrieved_q": retrieved_q, "retrieved_qid": retrieved_qid}


retrieval = Retrieval()
print(retrieval.retrieve("远洋夺宝是什么怎么玩"))
