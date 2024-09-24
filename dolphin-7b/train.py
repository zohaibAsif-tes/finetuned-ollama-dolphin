import json
import os
from pprint import pprint

import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    TextDataset,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Load Falcon Model and Tokenizer

MODEL_NAME = "cognitivecomputations/dolphin-2.0-mistral-7b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CASUAL_LM",
)

model = get_peft_model(model, config)


# Tokenize data

train_data = TextDataset(tokenizer, "./reach-chatbot.txt", block_size=128)


# Training

OUTPUT_DIR = "experiments"

training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=1,
    output_dir=OUTPUT_DIR,
    max_steps=80,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False

trainer.train()

# Save Trained Model

model.save_pretrained("reach_trained_dolphin_model")

print("Training complete and model saved successfully!")
