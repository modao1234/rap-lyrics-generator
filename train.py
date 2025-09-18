# train_rap_sft.py
from datasets import Dataset
import pandas as pd
import torch, os
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForSeq2Seq, TrainingArguments, Trainer
)
from peft import LoraConfig, TaskType, get_peft_model

MODEL_DIR = "./TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_JSONL = "./data/processed/lyrics.jsonl"
MAX_LENGTH = 512

SYSTEM_TXT = (
    "You are a rap lyrics generator. "
    "Follow the instruction (author, paragraph type, theme/mood, imagery, rhyme controls). "
    "Write exactly the requested lines with coherent rhyme and flow."
)

def process_func(example):
    sys = tokenizer(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_TXT}<|eot_id|>",
        add_special_tokens=False
    )
    user = tokenizer(
        f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}{example.get('input','')}<|eot_id|>",
        add_special_tokens=False
    )
    asst = tokenizer(
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{example['output']}<|eot_id|>",
        add_special_tokens=False
    )

    input_ids = sys["input_ids"] + user["input_ids"] + asst["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = sys["attention_mask"] + user["attention_mask"] + asst["attention_mask"] + [1]
    labels = [-100] * (len(sys["input_ids"]) + len(user["input_ids"])) + asst["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids       = input_ids[:MAX_LENGTH]
        attention_mask  = attention_mask[:MAX_LENGTH]
        labels          = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    df = pd.read_json(DATA_JSONL, lines=True)
    df = df[(df["instruction"].str.len() > 0) & (df["output"].str.len() > 0)].reset_index(drop=True)
    ds = Dataset.from_pandas(df)
    tokenized = ds.map(process_func, remove_columns=ds.column_names)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="./output/tinyllama_rap_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_steps=10,
        save_steps=100,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        save_on_each_node=True,
        remove_unused_columns=False
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
