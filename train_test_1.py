from torch import nn
import torch
from datasets import load_dataset
from trl import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer



BATCH_SIZE = 64

def get_model_and_tok():
    """Instantiating model and tokenizer"""
    model_name = "meta-llama/Llama-3.1-8B"

    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer



def get_dataset(dataset):
    """So each data point will be the max token length the model can handle unless the max will lead to a different pdf.

    
    """
    return dataset.map(preprocess_function, batched=True)


def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def get_trainer(model, tokenizer, dataset):
    """Setting up and instantiating Trainer"""

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp_16=True,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        #add datasets 


    )


if __name__ == "__main__":
    model, tokenizer = get_model_and_tok()

    dataset = get_dataset()

    trainer = get_trainer(model, tokenizer, dataset)

    trainer.train()

        




