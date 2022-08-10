# %%
from pathlib import Path
from typing import Union

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer

from uutils.torch_uu.data_uu.hf_uu_dataset import get_dataset_from_json_file, get_data_set_with_splits
from uutils.torch_uu.data_uu.hf_uu_tokenizer import re_train_tokenizer_from

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
tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = re_train_tokenizer_from(dataset, path2save_tokenizer=tokenizer_ckpt)

# - preprocess the data set ala HF trans tutorial (padding is not done here)

# - get model

# - get trainer, trainer args & it's data collate (padding is done here)

# - train it
