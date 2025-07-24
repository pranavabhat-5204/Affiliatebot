from transformers import pipeline, set_seed
import train_model.py

#Generating text using fine-tuned model
generator = pipeline('text-generation', model="./gpt2-title-desc", tokenizer=tokenizer)
set_seed(42)

title = "Best laptops"
output = generator(title, max_length=100, num_return_sequences=1)
print(output[0]['generated_text'])

# to load the RAG system based fine tuned model
from transformers import pipeline

generate_description('HP laptop')
