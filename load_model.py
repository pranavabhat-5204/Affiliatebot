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

generator = pipeline('text-generation', model="./gpt2-title-desc", tokenizer=tokenizer)

def generate_description(title):
    prompt = retrieve_context(title,0.3)
    prompt = f"Title:{title} \nDescription:{prompt}"
    output = generator(prompt, temperature=0.7, top_k=50, top_p=0.9, do_sample=True, max_length=100, num_return_sequences=1)
    return output[0]["generated_text"]
generate_description('HP laptop')
