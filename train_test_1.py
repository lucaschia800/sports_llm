from torch import nn
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from huggingface_hub import login
import os
from peft import LoraConfig, TaskType, get_peft_model

#login to huggingface

hf_token = os.getenv('HUGGINGFACE_TOKEN')

if hf_token:
    login(token=hf_token)  # Login with token
else:
    raise ValueError("HUGGINGFACE_TOKEN not found! Set it in your SLURM script.")

BATCH_SIZE = 2

def get_model_and_tok():
    """Instantiating model and tokenizer"""
    model_name = "meta-llama/Llama-3.1-8B"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_peft_model(model, config):

    return get_peft_model(model, config)



def load_datasets():
    """
    Loading in the datasets

    """
    ds = load_dataset('GEM/sportsett_basketball')
    ds_train = ds['train']
    ds_val = ds['validation']
    ds_test = ds['test']

    return ds_train, ds_val, ds_test


def preprocess_function(examples):
    #for now we are just tokenizing without padding but if it turns out llama doesn't accept then we will pad
    return tokenizer(examples["target"], truncation=True)


def process_dataset(ds):
    """Tokenizing and padding dataset"""
    ds = ds.map(preprocess_function, batched=True)
    return ds



def get_trainer(model, tokenizer, ds_train, ds_val):
    """Setting up and instantiating Trainer"""

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        tokenizer = tokenizer,
        train_dataset = ds_train,
        eval_dataset = ds_val,



    )

    return trainer


if __name__ == "__main__":
    model, tokenizer = get_model_and_tok()


    Lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'up_proj', 'down_proj'],  #start with training all change if overfitting
        r = 16,
        lora_alpha = 32, # typically double or equal to r, it is the lr scaling factor
        lora_droput = 0.01,
    )

    lora_model = get_peft_model(model, Lora_config)


    ds_train, ds_val, ds_test = load_datasets()

    ds_train_tokenized = process_dataset(ds_train)
    ds_val_tokenized = process_dataset(ds_val)

    

    trainer = get_trainer(lora_model, tokenizer, ds_train_tokenized, ds_val_tokenized)

    trainer.train()
