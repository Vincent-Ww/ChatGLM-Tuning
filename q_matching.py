# -- coding: utf-8 --
# @Author : wuzixun
# @Time : 2023/5/22 4:54 PM
# @File : q_matching

import pandas as pd
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from text_embedding import TextEmbedding


class Retrieval(object):
    def __init__(self):
        self.embedding = TextEmbedding()
        self.__init_vectordb()

    def __init_vectordb(self):
        part1 = pd.read_csv("data/h2h_question/part1.csv")
        part2 = pd.read_csv("data/h2h_question/part2.csv")
        combine = pd.concat([part1, part2], axis=0)
        h2h_q_names = combine['title'].to_list()[:2]
        h2h_q_ids = combine['qid'].to_list()[:2]
        questions = [Document(page_content=q, metadata={"qid": qid}) for q, qid in zip(h2h_q_names, h2h_q_ids)]
        self.db = Chroma(collection_name='h2h-questions', persist_directory='./chroma/biaozhunwen').from_documents(
            questions, embedding=self.embedding, persist_directory=".chroma/biaozhunwen")
        self.db.persist()

    def retrieve(self, query):
        retrived_res = self.db.as_retriever(search_type="similarity", search_kwargs={'k': 1}).get_relevant_documents(query)
        retrieved_q = retrived_res[0].page_content
        retrieved_qid = retrived_res.metadata['qid']
        return {"retrieved_q": retrieved_q, "retrieved_qid": retrieved_qid}


retrieval = Retrieval()
print(retrieval.retrieve("远洋夺宝是什么怎么玩"))
