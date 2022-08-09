#

## Language Model

LM = Language modeling predicts words in a sentence. 
It is one of several tasks you can formulate as a sequence-to-sequence problem, a powerful framework that extends to vision and audio tasks. 
https://huggingface.co/docs/transformers/tasks/language_modeling

There are two forms of LM:
- Causal language modeling predicts the next token in a sequence of tokens, and the model can only attend to tokens on the left.
  - https://huggingface.co/tasks/text-generation
  - notebooks from HF:
    - https://github.com/huggingface/blog/blob/main/notebooks/02_how_to_generate.ipynb
    - https://github.com/huggingface/blog/blob/main/notebooks/53_constrained_beam_search.ipynb
    - https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py
- Masked language modeling predicts a masked token in a sequence, and the model can attend to tokens bidirectionally.
  - https://huggingface.co/tasks/fill-mask
  - notebooks from HF
    - https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb
    - https://github.com/huggingface/notebooks/blob/main/examples/language_modeling_from_scratch.ipynb
    - https://github.com/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb

HF notesbooks: https://huggingface.co/docs/transformers/notebooks
HF code examples: https://github.com/huggingface/transformers/tree/main/examples/pytorch (seems less good?)


## Translation

Translation = Translation converts a sequence of text from one language to another. https://huggingface.co/docs/transformers/tasks/translation

- notebooks from HF:
  - https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb


### Previous model I trained as Enc-Dec arch/translation (tinfer)

- Prediction at training:
  - I gave the entire (right shifted) target as input to decoder (masks only for out of bounds, not for step of prediction)
  - (I did not loop through the len & masked the target input to dec nor the loss according to step)
- Prediction at testing (inference/eval):
  - I looped the length of the seq (i.e. auto-regressive) until a <eos> was predicted greedily.

- Q:
  - is this what T5 do?
  - is this what original trans do?
  - is this what T5/original trans do with HF?
