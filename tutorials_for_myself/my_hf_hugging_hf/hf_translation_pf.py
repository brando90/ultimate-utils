#%%

# https://huggingface.co/docs/transformers/tasks/translation
import datasets
from datasets import load_dataset

books = load_dataset("opus_books", "en-fr")
print(f'{books=}')

books = books["train"].train_test_split(test_size=0.2)

print(books["train"][0])
"""
{'id': '90560',
 'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
  'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}
"""

# - t5 tokenizer

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")

source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# def tokenize_function(examples: datasets.arrow_dataset.Batch):
#     encoded_batch: BatchEncoding = tokenizer(examples["text"], padding="max_length", truncation=True)
#     batch_size: int = len(examples['text'])
#     assert batch_size == len(examples['label'])
#     # encode_batch = tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")
#     # return tokenizer(examples["text"], padding="max_length", truncation=True)
#     # print(encoded_batch)
#     return encoded_batch  # e.g. {'input_ids': [[101, 173, 1197, 119, 22, ...}
#
# # use 🤗 Datasets map method to apply a preprocessing function over the entire dataset:
# tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=2)

tokenized_books = books.map(preprocess_function, batched=True, batch_size=2)

# - load model

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Use DataCollatorForSeq2Seq to create a batch of examples. It will also dynamically pad your text and labels to the
# length of the longest element in its batch, so they are a uniform length.
# While it is possible to pad your text in the tokenizer function by setting padding=True, dynamic padding is more efficient.

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

"""
At this point, only three steps remain:

Define your training hyperparameters in Seq2SeqTrainingArguments.
Pass the training arguments to Seq2SeqTrainer along with the model, dataset, tokenizer, and data collator.
Call train() to fine-tune your model.
"""

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()