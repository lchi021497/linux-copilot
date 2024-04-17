import os
import pdb
import torch
import json
import pandas as pd
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
from trl import DPOTrainer
from peft import PeftModel

#Global configs
model_name = "meta-llama/Llama-2-7b-chat-hf"
finetune_dir = "llama2-finetuned"
dpo_finetune_dir = 'llama2-dpo-finetuned'

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=MY_TOKEN)
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

model.config.use_cache = False
model.resize_token_embeddings(len(tokenizer))

# Load the adapter.
model = PeftModel.from_pretrained(
    model,
    finetune_dir,
    is_trainable=True,
    peft_config=peft_params,
    adapter_name="dpo_train",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)


# Load the adapter a second time, with a different name, which will be our reference model.
model.load_adapter(finetune_dir, adapter_name="reference")

if __name__ == "__main__":
    with open("./data/command_list.txt", "r") as rfile:
        commands = rfile.read().split("\n")[:-1]

    train_model = True
    if train_model:
        # dataset = load_dataset("json", data_files=f"./data/rlhf/answers1/dataset/alias_dataset.json")
        # pdb.set_trace()
        for cmd in tqdm(sorted(commands)):
            print("cmd ", cmd)
            if os.path.getsize(f"./data/rlhf/answers1/dataset/{cmd}_dataset.json") < 1000:
                continue

            with open(f"./data/rlhf/answers1/dataset/{cmd}_dataset.json", "r") as rfile:
                json_dataset= json.load(rfile)
                dataset = Dataset.from_dict(json_dataset)
            
            dpo_trainer = DPOTrainer(
                model,
                train_dataset=dataset,
                tokenizer=tokenizer,
                model_adapter_name="dpo_train",
                ref_adapter_name=None,
                beta=0.1,
                args=training_params
            )

            dpo_trainer.train()
        
            dpo_trainer.model.save_pretrained(dpo_finetune_dir, save_adapter=True, save_config=True)
