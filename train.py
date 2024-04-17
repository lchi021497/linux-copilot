import os
import pandas as pd
import pyarrow as pa
import pdb
import torch
from common import load_df_dataset, MY_TOKEN, peft_params
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Configs
model_name = "meta-llama/Llama-2-7b-chat-hf"
finetuned_dir = "llama2-finetuned"

cmd_desc_dir = "data/sft/description.pkl"
rev_cmd_desc_dir = "data/sft/rev_description.pkl"
options_dir = "data/sft/options.pkl"


#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=MY_TOKEN, download_mode="force_redownload")
#Create a new token and add it to the tokenizer
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.padding_side = 'right'

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
          model_name, quantization_config=bnb_config, device_map={"": 0}, token=MY_TOKEN
)
# model = PeftModel(model, peft_config=peft_params)

# TODO: check prepare_model_for_int8_training
#Resize the embeddings
model.resize_token_embeddings(len(tokenizer))
#Configure the pad token in the model
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching

desc_dataset = load_df_dataset(cmd_desc_dir, load_type="train")
rev_desc_dataset = load_df_dataset(rev_cmd_desc_dir, load_type="train")
options_dataset = load_df_dataset(options_dir, load_type="train")

dataset = pd.concat((desc_dataset, rev_desc_dataset, options_dataset)).sample(frac=1)
pd.set_option('display.max_colwidth', None)
print(options_dataset.iloc[0])
dataset = Dataset.from_pandas(dataset)
test_desc_dataset = Dataset.from_pandas(load_df_dataset(cmd_desc_dir, load_type="infer"))
# dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=1000,
    logging_steps=1000,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()
# pdb.set_trace()
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(test_desc_dataset[0]['text'])
print(result[0]['generated_text'])
 
trainer.model.save_pretrained(finetuned_dir, save_adapter=True, save_config=True)
