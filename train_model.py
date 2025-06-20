!pip install pandas
!pip install --upgrade pandas datasets huggingface_hub
!pip install --upgrade transformers
!pip install torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ds = load_dataset("Studeni/AMAZON-Products-2023")
dataset={'title':ds['train']['title'][:10], 'description':ds['train']['description'][:10]}
df={}
df['text']=[str(dataset['title'][i])+str(dataset['description'][i]) for i in range(len(dataset['title']))]
data=pd.DataFrame(df)

model_name = "gpt2" # You can change this to another model like "gpt2-medium", "gpt2-large", etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

tokenized_dataset = []
for a in data['text']:
  tokenized_dataset.append(tokenizer(a))
tokenizer.pad_token = tokenizer.eos_token

training_args = TrainingArguments(
    output_dir="./gpt2-title-desc",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=50,
    save_steps=200,
    eval_strategy="no",
    warmup_steps=100,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model()

#Now an RAG system to retrieve data from additional information
dataset={'title':ds['train']['title'][11:], 'description':ds['train']['description'][11:]}
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
corpus=[]
i=11
while i<len(dataset['title']):
  corpus.append(str(dataset['title'][i])+' '+str(dataset['description'][i]))
  i=i+1

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
def retrieve_context(query,threshold):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, X)
    top_score = scores.max()
    top_idx = scores.argmax()
    # Retrieve the title and description using the top_idx
    # Combine them to form the context
    context = f"Title: {dataset['title'][top_idx]} Description: {dataset['description'][top_idx]}"
    if top_score >= threshold:
        context = corpus[top_idx]
        prompt = f"Title: {title}\nRelevant Info: {context}\nDescription:"
    else:
        prompt = f"Title: {title}\nDescription:"

    return prompt
  
from transformers import pipeline

generator = pipeline('text-generation', model="./gpt2-title-desc", tokenizer=tokenizer)

def generate_description(title):
    prompt = retrieve_context(title,0.3)
    prompt = f"Title:{title} \nDescription:{prompt}"
    output = generator(prompt, temperature=0.7, top_k=50, top_p=0.9, do_sample=True, max_length=100, num_return_sequences=1)
    return output[0]["generated_text"]
generate_description('HP laptop')
