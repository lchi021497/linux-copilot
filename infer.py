import os
import pdb
import torch
from tqdm import tqdm
from common import load_df_dataset, MY_TOKEN, peft_params
from datasets import load_dataset, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import PeftModel

# global configs
model_name = "meta-llama/Llama-2-7b-chat-hf"
#adapter_name = "archive/llama2-dpo-finetuned/dpo_train"
adapter_name = "lchi37/linux-copilot"
infer_model_dir = "llama_infer_model"

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
infer_model = AutoModelForCausalLM.from_pretrained(
          model_name, quantization_config=bnb_config, device_map={"": 0}, token=MY_TOKEN
)
# print(infer_model.layers[0].weight)
# for name, weights in infer_model.named_parameters():
#     print(name, weights.shape)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=MY_TOKEN)
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.padding_side = 'right'

infer_model.resize_token_embeddings(len(tokenizer))

infer_model = PeftModel.from_pretrained(infer_model, adapter_name)

infer_model = infer_model.merge_and_unload()
infer_pipe = pipeline(task="text-generation", model=infer_model, tokenizer=tokenizer, max_length=1500)

save_model = True
if save_model:
    print("saving model")
    infer_model.save_pretrained(infer_model_dir)

if __name__ == "__main__":
    rev_test_desc_dataset = Dataset.from_pandas(load_df_dataset("data/sft/description.pkl", load_type="infer"))
    
    for test_desc in tqdm(rev_test_desc_dataset):
        result = infer_pipe(test_desc["text"])
        print(result[0]['generated_text'])
