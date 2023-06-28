"""
Accelerate is a library that enables the same PyTorch code to be run across any distributed configuration by adding just
four lines of code!

tldr; handles all from cpu-gpu(s)-multi-node-tpu-tpu + deepseed + mixprecision in one simple wrapper without complicated
calls e.g. that ddp has to do for multi gpus.

ref: my notes: https://www.evernote.com/shard/s410/sh/f1158fa5-4122-0d17-d6eb-a920461e12b6/g47Qtu6j1F58zvMnJ3fWY8v6pFFWi3I_krn5155UigRUmBzr-D8td5HaQA

later, write 1 so question confirm code bellow:
https://discuss.huggingface.co/t/trainer-and-accelerate/26382
https://stackoverflow.com/questions/ask
"""
#%%
# + from accelerate import Accelerator
# + accelerator = Accelerator()
#
# + model, optimizer, training_dataloader, scheduler = accelerator.prepare(
# +     model, optimizer, training_dataloader, scheduler
# + )
#
#   for batch in training_dataloader:
#       optimizer.zero_grad()
#       inputs, targets = batch
#       inputs = inputs.to(device)
#       targets = targets.to(device)
#       outputs = model(inputs)
#       loss = loss_function(outputs, targets)
# +     accelerator.backward(loss)
#       optimizer.step()
#       scheduler.step()

# ref: https://chat.openai.com/share/1014a48a-d714-472f-9285-d6baa419fe6b

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AdamW
from accelerate import Accelerator
from datasets import load_dataset
import torch

# Initialize accelerator
accelerator = Accelerator()

# Load a dataset
dataset = load_dataset('text', data_files={'train': 'train.txt', 'test': 'test.txt'})

# Tokenization
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

def tokenize_function(examples):
    # We are doing causal (unidirectional) masking
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# Split the dataset into train and test
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Initialize model
model = GPT2LMHeadModel.from_pretrained("gpt2")
optimizer = AdamW(model.parameters())

# Prepare everything with our `accelerator`.
model, optimizer, train_dataset, test_dataset = accelerator.prepare(model, optimizer, train_dataset, test_dataset)

# Now let's define our training loop
device = accelerator.device
model.train()

for epoch in range(3):
    for step, batch in enumerate(train_dataset):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    # Evaluation logic
    model.eval()
    eval_loss = 0.0
    eval_steps = 0

    for batch in test_dataset:
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            eval_loss += outputs.loss.item()
        eval_steps += 1

    eval_loss = eval_loss / eval_steps
    print(f'Evaluation loss: {eval_loss}')

    model.train()


#%%
"""
ref: https://chat.openai.com/share/1014a48a-d714-472f-9285-d6baa419fe6b
"""
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from accelerate import Accelerator
from datasets import load_dataset

# Initialize accelerator
accelerator = Accelerator()

# Load a dataset
dataset = load_dataset('imdb')

# Tokenization
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into train and test
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Initialize model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Prepare everything with our `accelerator`.
model, train_dataset, test_dataset = accelerator.prepare(model, train_dataset, test_dataset)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir='./logs',
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()
# %%
"""
ref: https://chat.openai.com/share/1014a48a-d714-472f-9285-d6baa419fe6b
"""
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TrainingArguments, Trainer
from accelerate import Accelerator
from datasets import load_dataset

# Initialize accelerator
accelerator = Accelerator()

# Load a dataset
dataset = load_dataset('text', data_files={'train': 'train.txt', 'test': 'test.txt'})

# Tokenization
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

def tokenize_function(examples):
    # We are doing causal (unidirectional) masking
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set the columns to be used in training
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask"])

# Split the dataset into train and test
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Initialize model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Prepare everything with our `accelerator`.
model, train_dataset, test_dataset = accelerator.prepare(model, train_dataset, test_dataset)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    prediction_loss_only=True,  # In language modelling, we only care about the loss
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()
