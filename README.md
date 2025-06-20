# Affiliatebot
This is a generative AI model for Affiliate marketers which is a fine-tuned LLM ( with amazon product data) to help them with answering queries of their customers. It has RAG implementation with the model in order to give the flexibility to add more recent data.

The needed dependencies are:
* pandas
* torch
* transformers
* datasets

The steps involved in this project are:
The code first installs necessary libraries like `PyMuPDF`, `pandas`, `os`, `re`, `fsspec`, `torch`, `transformers`, and `datasets`.
*   It loads the "Studeni/AMAZON-Products-2023" dataset using the `datasets` library.
*   It extracts the 'title' and 'description' columns for the first 10000 entries and creates a pandas DataFrame. This is because of the memory limitations of the device and so this was one way to complete the fine-tuning.
*   It loads the pre-trained `gpt2` model and its corresponding tokenizer using the `transformers` library .
*   Then we implement code to actually generate descriptions using the loaded model.
*   This fine-tuning of the GPT-2 model on the Amazon product description data gives better results for product based queries better results.
*   Now in order to use the remaining part of the dataset, we use RAG to use that corpus of data.
*   Here we use the TF-IDF method.
*   I have improvised this by adding a threshold score, so as to not taking prompts that do not match enough.
*   After this the model is able to give out really relevant results.

