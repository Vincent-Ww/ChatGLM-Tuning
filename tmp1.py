# -- coding: utf-8 --
# @Author : wuzixun
# @Time : 2023/7/3 10:57 AM
# @File : tmp1

## 将送标的评估数据用SFT-LLM跑一遍，找出不一样的数据。

import pandas as pd
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from tqdm import tqdm
import json
import xlsxwriter


def chatglm_inference(model, tokenizer, sample):
    context = f"Instruction: {sample['instruction']}\n"
    context += f"Input: {sample['input']}\n"
    context += "Answer: "

    response, _ = model.chat(tokenizer, context, history=[], do_sample=False)
    return response


# def to_annotation_format(json_data, output_path):
#     workbook = xlsxwriter.Workbook(output_path)
#     worksheet = workbook.add_worksheet()
#     bold_red = workbook.add_format({'bold': True, 'color': 'red'})
#     wrap = workbook.add_format({'text_wrap': True})
#     sheet = workbook.add_worksheet()
#     worksheet.write(0, 0, "sessionid")
#     worksheet.write(0, 1, "dialogue")
#     worksheet.write(0, 2, "AI一级FT")
#     worksheet.write(0, 3, "AI二级FT")
#     worksheet.write(0, 4, "是否转接")
#     worksheet.write(0, 5, "首次关联工单功能树")
#     # worksheet.write(0, 6, "session FT(待标注)")
#     worksheet.write(0, 6, "SessionFT-chatglm")
#     nrow = 1
#     for ele in json_data:
#         worksheet.write(nrow, 0, ele['session_id'])
#
#         dialogue = ele['input']
#         splits = dialogue.split("\n")
#         dialogue_input = []
#         for sentence in splits:
#             if len(sentence) == 0:
#                 continue
#             if sentence.startswith("用户:"):
#                 dialogue_input.append(bold_red)
#             dialogue_input.append(sentence + "\n")
#         worksheet.write_rich_string(nrow, 1, *dialogue_input, wrap)
#
#         ft1 = ele['AI一级FT'] if isinstance(ele['AI一级FT'], str) else ""
#         worksheet.write(nrow, 2, ft1)
#         ft2 = ele['AI二级FT'] if isinstance(ele['AI二级FT'], str) else ""
#         worksheet.write(nrow, 3, ft2)
#         worksheet.write(nrow, 4, ele['是否转接'])
#         worksheet.write(nrow, 5, ele['output'])
#         worksheet.write(nrow, 6, chatglm_inference(model, tokenizer, ele))
#         nrow += 1
#
#     workbook.close()

PEFT_PATH = "/home/xiezizhe/wuzixun/LLM/ChatGLM-Tuning/ks-ai-ft5-20230626"
CHATGLM_PATH = "/home/xiezizhe/wuzixun/LLM/chatglm-6b"
tokenizer = AutoTokenizer.from_pretrained(CHATGLM_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(CHATGLM_PATH, load_in_8bit=True, trust_remote_code=True, device_map='auto')
model = model.eval()
model = PeftModel.from_pretrained(model, PEFT_PATH)
if __name__ == "__main__":
    file_path = "eval_1000_beforeAnno.xlsx"
    eval_data = pd.read_excel(file_path, sheet_name="Sheet1")
    chatglm_result_list = []
    llm_1ft_list = []
    llm_2ft_list = []
    for i in range(eval_data.shape[0]):
        row = eval_data.iloc[i]
        sample = {"instruction": "下面是用户和客服的一段对话，请你根据下面这段对话总结出所属功能树。\n功能树是区分用户诉求类别的多层树状知识结构，最多只有两个层级，不同层级之间用\"~\"相连。(例如 电商~买家, 平台~账号)。",
                  "input": row['dialogue']}
        chatglm_result = chatglm_inference(model, tokenizer, sample)
        chatglm_result_list.append(chatglm_result)
        llm_1ft_list.append(chatglm_result.split("~")[0])
        llm_2ft_list.append(chatglm_result.split("~") if "~" in chatglm_result else "无")

    eval_data['LLM结果'] = chatglm_result_list
    eval_data['LLM一级FT'] = llm_1ft_list
    eval_data['LLM二级FT'] = llm_2ft_list
    eval_data.to_excel("sft_compare.xlsx")



