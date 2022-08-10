# %%
# - get our custom data set (load from file)
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

# - get our t5 tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("t5-small")
print(f'{isinstance(tokenizer, PreTrainedTokenizer)=}')
print(f'{isinstance(tokenizer, PreTrainedTokenizerFast)=}')

# modify this to preprocess our data correctly
from uutils.torch_uu.data_uu.hf_uu_data_preprocessing import helper_get_preprocess_function_translation_tutorial

preprocessor = helper_get_preprocess_function_translation_tutorial(tokenizer)
tokenized_books = books.map(preprocessor, batched=True, batch_size=2)

# - load a t5 model
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Use DataCollatorForSeq2Seq to create a batch of examples. It will also dynamically pad your text and labels to the
# length of the longest element in its batch, so they are a uniform length.
# While it is possible to pad your text in the tokenizer function by setting padding=True, dynamic padding is more efficient.
from transformers import DataCollatorForSeq2Seq
# Data collator that will dynamically pad the inputs received, as well as the labels.
data_collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)