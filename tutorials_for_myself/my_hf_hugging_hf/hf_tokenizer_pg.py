"""
ref: https://huggingface.co/docs/transformers/v4.21.1/en/fast_tokenizers
"""
#%%

# We now have a tokenizer trained on the files we defined. We can either continue using it in that runtime, or save it to a JSON file for future re-use.
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer.pre_tokenizer = Whitespace()
files = [...]
tokenizer.train(files, trainer)

# Loading directly from the tokenizer object

from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

# Loading from a JSON file

tokenizer.save("tokenizer.json")

from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")