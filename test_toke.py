from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset




ds = Dataset.from_dict({'text': [('Also, using batched processing might be more efficient. Wait, if the user processes examples in batches, can we handle that? Maybe not, because each text in the batch might have different lengths. So processing one example at a time (batched=False) might be easier, even if less efficient. But for large datasets, this could be slow. However, given that the user is dealing with continued pre-training, which often involves large datasets, efficiency is important.'
                                  'Also, using batched processing might be more efficient. Wait, if the user processes examples in batches, can we handle that? Maybe not, because each text in the batch might have different lengths. So processing one example at a time (batched=False) might be easier, even if less efficient. But for large datasets, this could be slow. However, given that the user is dealing with continued pre-training, which often involves large datasets, efficiency is important.'
                                    'Also, using batched processing might be more efficient. Wait, if the user processes examples in batches, can we handle that? Maybe not, because each text in the batch might have different lengths. So processing one example at a time (batched=False) might be easier, even if less efficient. But for large datasets, this could be slow. However, given that the user is dealing with continued pre-training, which often involves large datasets, efficiency is important.'
                                    'Also, using batched processing might be more efficient. Wait, if the user processes examples in batches, can we handle that? Maybe not, because each text in the batch might have different lengths. So processing one example at a time (batched=False) might be easier, even if less efficient. But for large datasets, this could be slow. However, given that the user is dealing with continued pre-training, which often involves large datasets, efficiency is important.'
                                    'Also, using batched processing might be more efficient. Wait, if the user processes examples in batches, can we handle that? Maybe not, because each text in the batch might have different lengths. So processing one example at a time (batched=False) might be easier, even if less efficient. But for large datasets, this could be slow. However, given that the user is dealing with continued pre-training, which often involves large datasets, efficiency is important.'
                                    'Also, using batched processing might be more efficient. Wait, if the user processes examples in batches, can we handle that? Maybe not, because each text in the batch might have different lengths. So processing one example at a time (batched=False) might be easier, even if less efficient. But for large datasets, this could be slow. However, given that the user is dealing with continued pre-training, which often involves large datasets, efficiency is important.'
                                    'Also, using batched processing might be more efficient. Wait, if the user processes examples in batches, can we handle that? Maybe not, because each text in the batch might have different lengths. So processing one example at a time (batched=False) might be easier, even if less efficient. But for large datasets, this could be slow. However, given that the user is dealing with continued pre-training, which often involves large datasets, efficiency is important.'
                                    'Also, using batched processing might be more efficient. Wait, if the user processes examples in batches, can we handle that? Maybe not, because each text in the batch might have different lengths. So processing one example at a time (batched=False) might be easier, even if less efficient. But for large datasets, this could be slow. However, given that the user is dealing with continued pre-training, which often involves large datasets, efficiency is important.')]})

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
print(tokenizer.model_max_length)
tokenizer.pad_token = tokenizer.eos_token

ds_tokenized = ds.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True), batched=True)


print(ds_tokenized[0].keys())
