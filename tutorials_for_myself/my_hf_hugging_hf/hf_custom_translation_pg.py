# %%
# - get the dataset
from pathlib import Path
from typing import Union

from datasets import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from uutils.torch_uu.data_uu.hf_uu_dataset import get_dataset_from_json_file
from uutils.torch_uu.data_uu.hf_uu_tokenizer import re_train_tokenizer_from

path: Path = Path('~/data/pycoq_lf_debug/').expanduser()
dataset: Dataset = get_dataset_from_json_file(path / '/flatten_data.json')

# - get our tokenizer
tokenizer_ckpt: Path = path / 'tokenizer'
# tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = AutoTokenizer.from_pretrained(tokenizer_ckpt)
tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = re_train_tokenizer_from(dataset)

# - preprocess the data set ala HF trans tutorial

# - get model

# - get trainer, trainer args & it's data collate

# - train it
