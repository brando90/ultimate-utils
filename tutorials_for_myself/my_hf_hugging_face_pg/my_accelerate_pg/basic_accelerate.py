"""
Accelerate is a library that enables the same PyTorch code to be run across any distributed configuration by adding just
four lines of code!
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
