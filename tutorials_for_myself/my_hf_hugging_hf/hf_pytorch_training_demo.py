"""
ref: https://huggingface.co/docs/transformers/training#train-in-native-pytorch
"""
#%%

# - Prepare a dataset
import datasets
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
print(f'{dataset["train"]=}')
dataset["train"][100]
print(f'print data point: {dataset["train"][100]=}')
'''
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
'''

# - tokenize
from transformers import AutoTokenizer, BatchEncoding

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples: datasets.arrow_dataset.Batch):
    encoded_batch: BatchEncoding = tokenizer(examples["text"], padding="max_length", truncation=True)
    batch_size: int = len(examples['text'])
    assert batch_size == len(examples['label'])
    # encode_batch = tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")
    # return tokenizer(examples["text"], padding="max_length", truncation=True)
    # print(encoded_batch)
    return encoded_batch  # e.g. {'input_ids': [[101, 173, 1197, 119, 22, ...}

# use 🤗 Datasets map method to apply a preprocessing function over the entire dataset:
tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=2)
print(tokenized_datasets)

# Next, manually postprocess tokenized_dataset to prepare it for training.
# Remove the text column because the model does not accept raw text as an input:
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

#Rename the label column to labels because the model expects the argument to be named labels:
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Set the format of the dataset to return PyTorch tensors instead of lists:
tokenized_datasets.set_format("torch")

# Then create a smaller subset of the dataset as previously shown to speed up the fine-tuning:
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# - Create a DataLoader for your training and test datasets so you can iterate over batches of data:
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=4)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=2)

# - Load your model with the number of expected labels:
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

# - get pytorch optimiser
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# - Create the default learning rate scheduler from (HF) Trainer:
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# - to gpu
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# - training loop: https://huggingface.co/docs/transformers/training#training-loop
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# - Just like how you need to add an evaluation function to Trainer, you need to do the same when you write your own training loop. But instead of calculating and reporting the metric at the end of each epoch, this time you will accumulate all the batches with add_batch and calculate the metric at the very end.
from datasets import load_metric

metric = load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()