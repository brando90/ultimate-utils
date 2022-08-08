#%%
"""
Goal: use a standard off-the-shelf tokenizer (i.e. HuggingFace BPE) but
    then re-train it on the statistics of your code dataset.

refs:
- https://huggingface.co/docs/transformers/v4.21.1/en/fast_tokenizers
"""

# from transformers import AutoTokenizer, XLMRobertaTokenizer
from pathlib import Path

from transformers import AutoTokenizer

path: Path = Path('~/data/tmp/').expanduser()
path.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
assert tokenizer.is_fast
print(tokenizer)

# - save tokenizer
tokenizer.save_pretrained(path / "pre-train-roberta-base")
tokenizer.save_pretrained("pre-train-roberta-base")

# - try out tokenizer
tokenizer_checkpoint = "nasa-tokenizer"
# tokenizer_checkpoint = path / "pre-train-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

#%%
