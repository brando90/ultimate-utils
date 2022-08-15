#%%
# https://huggingface.co/docs/transformers/training, https://huggingface.co/docs/transformers/training#train

# - Prepare a dataset
import datasets
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
print(f'{dataset=}')
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

# If you like, you can create a smaller subset of the full dataset to fine-tune on to reduce the time it takes:
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# 🤗 Transformers provides a Trainer class optimized for training 🤗 Transformers models, making it easier to start training without manually writing your own training loop.
# The Trainer API supports a wide range of training options and features such as logging, gradient accumulation, and mixed precision.

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
# You will see a warning about some of the pretrained weights not being used and some weights being randomly initialized.
# Don’t worry, this is completely normal! The pretrained head of the BERT model is discarded, and replaced with a randomly initialized classification head.
# You will fine-tune this new model head on your sequence classification task, transferring the knowledge of the pretrained model to it.

# Training hyperparameters
from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")

# Metric
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# If you’d like to monitor your evaluation metrics during fine-tuning, specify the evaluation_strategy parameter in your
# training arguments to report the evaluation metric at the end of each epoch:
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# Create a Trainer object with your model, training arguments, training and test datasets, and evaluation function:
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()


