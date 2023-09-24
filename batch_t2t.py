from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
model_name = 'google/flan-t5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

texts = ["translate English to German: How old are you?", "tell me what google is", "how are you doing"]

for input_text in texts:
    input_ids = tokenizer(input_text, return_tensors="pt", max_length = 512, padding = True).input_ids
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))