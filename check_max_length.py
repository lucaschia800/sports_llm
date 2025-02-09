from torch import nn
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

max_length = tokenizer.model_max_length

print(max_length)