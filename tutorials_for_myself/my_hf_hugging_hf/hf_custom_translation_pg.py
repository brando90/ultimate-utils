# %%
from pathlib import Path
from typing import Union

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

from uutils.torch_uu.data_uu.hf_uu_dataset import get_dataset_from_json_file, get_data_set_with_splits
from uutils.torch_uu.data_uu.hf_uu_tokenizer import re_train_tokenizer_from

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

import torch

# - get the dataset
path: Path = Path('~/data/pycoq_lf_debug/').expanduser()
dataset: Dataset = get_dataset_from_json_file(path / 'flatten_data.jsonl')
print(f'{dataset=}')
print(f'{dataset[0]=}')

dataset: DatasetDict = get_data_set_with_splits(dataset)
print(f'{dataset["train"]=}')
print(f'{dataset["train"][0]=}')

# - get our tokenizer
# todo: have code to pre-train the tokenizer
tokenizer_ckpt: Path = path / 'tokenizer'
# if tokenizer_ckpt.exists():
#     tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = AutoTokenizer.from_pretrained(tokenizer_ckpt)
# else:
#     tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = re_train_tokenizer_from(dataset,
#                                                                                              path2save_tokenizer=tokenizer_ckpt)
tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = re_train_tokenizer_from(dataset,
                                                                                         path2save_tokenizer=tokenizer_ckpt)

# - preprocess the data set ala HF trans tutorial (padding is not done here)
from data_pkg.data_preprocessing import get_preprocessed_tokenized_datasets

dataset: DatasetDict = get_preprocessed_tokenized_datasets(dataset, tokenizer, batch_size=2)

# - get model

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# - get trainer, trainer args & it's data collate (padding is done here)

from transformers import DataCollatorForSeq2Seq

# Data collator that will dynamically pad the inputs received, as well as the labels.
data_collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

"""
At this point, only three steps remain:

- Define your training hyperparameters in Seq2SeqTrainingArguments.
- Pass the training arguments to Seq2SeqTrainer along with the model, dataset, tokenizer, and data collator.
- Call train() to fine-tune your model.
"""

fp16: bool = torch.cuda.is_available()  # True for cuda, false for cpu
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",  # todo change
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
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# - train it
