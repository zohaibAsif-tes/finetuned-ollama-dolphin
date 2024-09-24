import os

import bitsandbytes as bnb
import torch
import transformers

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    TextDataset,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# This specifies that only the first GPU (device 0) should be used for training.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Model Name: "cognitivecomputations/dolphin-2.0-mistral-7b" is the base model used for fine-tuning.
MODEL_NAME = "cognitivecomputations/dolphin-2.0-mistral-7b"

# The BitsAndBytesConfig configures the model to use 4-bit precision, which significantly reduces the memory footprint
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Loads the model for causal language modeling
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# This enables gradient checkpointing to reduce memory usage by trading off some computational overhead
model.gradient_checkpointing_enable()

# Prepares the model for 4-bit training, ensuring that only a subset of parameters is trained efficiently
model = prepare_model_for_kbit_training(model)

#  LoRA config for model quantization
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CASUAL_LM",
)

# Integrates the LoRA configuration into the model for fine-tuning
model = get_peft_model(model, config)


# A text dataset is created from reach-chatbot.txt. 
# The block_size=128 controls the maximum sequence length per training sample
train_data = TextDataset(tokenizer, "./reach-chatbot.txt", block_size=128)


# Training

OUTPUT_DIR = "experiments"

# Set training arguments
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

# The Trainer class from Hugging Face is used to manage the training loop
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Disables caching of previous outputs during training to save memory
model.config.use_cache = False

# Starts the training process
trainer.train()

# Save Trained Model
model.save_pretrained("reach_trained_dolphin_model")

print("Training complete and model saved successfully!")
