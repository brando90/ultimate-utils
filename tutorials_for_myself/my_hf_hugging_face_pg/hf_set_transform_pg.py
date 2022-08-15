"""
I think this will be useful to me for:
- doing vision, where transforms on the fly are needed
- ml4tp, where new task creation on the fly from raw data is possible (where all data is not always available in the
    hf preprocessing like it is in nl).

- https://huggingface.co/docs/datasets/package_reference/main_classes?highlight=set_transform#datasets.Dataset.set_transform
"""
#%%

from datasets import load_dataset
from transformers import AutoTokenizer
ds = load_dataset("rotten_tomatoes", split="validation")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def encode(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')
ds.set_transform(encode)
ds[0]