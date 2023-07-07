"""

ref: https://chat.openai.com/share/0bcaf2cb-5305-4716-b7d5-7c8d0922171e
ref: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
"""

# todo: make into funcs + debug + check with john's code
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Expand vocabulary with new tokens
new_tokens = ['Aragorn', 'Frodo', 'Lothlorien']
num_new_tokens = tokenizer.add_tokens(new_tokens)

# Initialize model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Resize the token embeddings
model.resize_token_embeddings(len(tokenizer))

# Calculate the average embedding to initialize new embeddings
avg_embedding = model.get_input_embeddings().weight[:model.config.vocab_size-num_new_tokens, :].mean(dim=0)

# Initialize new embeddings with the average embedding
with torch.no_grad():
    model.get_input_embeddings().weight[-num_new_tokens:] = avg_embedding

# Now, you can fine-tune your model on the new data
