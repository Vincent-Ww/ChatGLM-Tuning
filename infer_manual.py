# -- coding: utf-8 --
# @Author : wuzixun
# @Time : 2023/8/14 10:59 AM
# @File : infer_manual.py

import sys
import argparse
import time
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


def chatglm_inference(model, tokenizer, prompt):
    response, _ = model.chat(tokenizer, prompt, history=[], do_sample=False)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatglm_path", type=str, default="/home/xiezizhe/wuzixun/LLM/chatglm-6b")
    parser.add_argument("--peft_path", type=str, default="ks-ai-ft-20230807")
    FLAGS, _ = parser.parse_known_args()
    chatglm_path = FLAGS.chatglm_path
    peft_path = "/home/xiezizhe/wuzixun/LLM/ChatGLM-Tuning/" + FLAGS.peft_path

    tokenizer = AutoTokenizer.from_pretrained(chatglm_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(chatglm_path, trust_remote_code=True, device_map='auto').half()
    model = model.eval()

    model = PeftModel.from_pretrained(model, peft_path)

    while True:
        print("================")
        # prompt = sys.stdin.readline().strip('\n')

        prompt = []
        while True:
            aLine = sys.stdin.readline()
            if "endendend" in aLine:
                break
            prompt.append(aLine)
        prompt = "\n".join(prompt)

        start = time.time()
        answer = chatglm_inference(model, tokenizer, prompt)
        end = time.time()

        print(answer)
        print(end - start)
