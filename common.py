import os
import pandas as pd
from datasets import Dataset
from peft import LoraConfig

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
)


MY_TOKEN = os.environ["huggingface_token"]


def format_train_example(question, answer):
    train_template = """<s>[INST] <<SYS>>
    You are a helpful, respectful and honest code assistant. Always answer as helpfully as possible, while being safe. Your answers should be concise when asked what command does what, starting with answer on the first line, followed by an explanation of the steps you took to arrive at that answer.
   
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>
   
    {} [/INST] {} </s>"""
   
    return train_template.format(question, answer)

def format_test_example(question):
    infer_template = """<s>[INST] <<SYS>>
    You are a helpful, respectful and honest code assistant. Always answer as helpfully as possible, while being safe. Your answers should be concise when asked what command does what, starting with answer on the first line, followed by an explanation of the steps you took to arrive at that answer.
   
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>
   
    {} [/INST]"""
   
    return infer_template.format(question)

def extract_result(answer):
    inst_label = "[/INST]"
    return answer[answer.find(inst_label) + len(inst_label):]

def load_df_dataset(fname, load_type="train"):
    df = pd.read_pickle(fname)
    
    assert load_type == "train" or load_type == "infer"
    prepared_df = None
    if load_type == "train":
        prepared_df = df.apply(lambda x: format_train_example(x["question"], x["answer"]), axis=1).to_frame()
    else:
        prepared_df = df.apply(lambda x: format_test_example(x["question"]), axis=1).to_frame()
    
    prepared_df.columns = ["text"]
    return prepared_df 
