# -- coding: utf-8 --
# @Author : wuzixun
# @Time : 2023/5/26 10:28 AM
# @File : retrieve_top5

import pandas as pd
from q_matching import Retrieval

ret = Retrieval(load_from_disk=True, persist_directory=".chroma/biaozhunwen")

data = pd.read_excel("experiments/人人对话LLM验证5800_匹配人工Q.xlsx")

top5_list = []
top5_recall = []
for i in range(data.shape[0]):
    llm_answer = data.iloc[i]['LLM结果(微调后)']
    top5 = ret.retrieve(llm_answer, topK=5)
    label = data.iloc[i]['标准问']
    top5_list.append("\n".join(top5))
    top5_recall.append(label in top5)
data['检索top5'] = top5_list
data['top5是否命中'] = top5_recall
data.to_excel("experiments/人人对话LLM验证5800_匹配人工Q_top5.xlsx")
