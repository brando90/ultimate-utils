#%%

# https://huggingface.co/docs/transformers/tasks/translation
import datasets
from datasets import load_dataset, DatasetDict

books: DatasetDict = load_dataset("opus_books", "en-fr")
print(f'{books=}')

books: DatasetDict = books["train"].train_test_split(test_size=0.2)
print(f'{books=}')

print(books["train"][0])
"""
{'id': '90560',
 'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
  'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}
"""

# - t5 tokenizer

from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("t5-small")
print(f'{isinstance(tokenizer, PreTrainedTokenizer)=}')
print(f'{isinstance(tokenizer, PreTrainedTokenizerFast)=}')

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


# # use 🤗 Datasets map method to apply a preprocessing function over the entire dataset:
# tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=2)

# todo - would be nice to remove this since gpt-2/3 size you can't preprocess the entire data set...or can you?
# tokenized_books = books.map(preprocess_function, batched=True, batch_size=2)
from uutils.torch_uu.data_uu.hf_uu_data_preprocessing import helper_get_preprocess_function_translation_tutorial
preprocessor = helper_get_preprocess_function_translation_tutorial(tokenizer)
tokenized_books = books.map(preprocessor, batched=True, batch_size=2)
print(f'{tokenized_books=}')

# - load model
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Use DataCollatorForSeq2Seq to create a batch of examples. It will also dynamically pad your text and labels to the
# length of the longest element in its batch, so they are a uniform length.
# While it is possible to pad your text in the tokenizer function by setting padding=True, dynamic padding is more efficient.

from transformers import DataCollatorForSeq2Seq

# Data collator that will dynamically pad the inputs received, as well as the labels.
data_collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

"""
At this point, only three steps remain:

- Define your training hyperparameters in Seq2SeqTrainingArguments.
- Pass the training arguments to Seq2SeqTrainer along with the model, dataset, tokenizer, and data collator.
- Call train() to fine-tune your model.
"""

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# fp16 = True # cuda
#fp16 = False # cpu
import torch
fp16 = torch.cuda.is_available()  # True for cuda, false for cpu
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    fp16=fp16,
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