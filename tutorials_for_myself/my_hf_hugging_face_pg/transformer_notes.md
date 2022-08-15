# The Ultimate Transformer notes

## Foundations of Foundation Model (pun intended)

Transformers take the entire sequence as input and therefore do not suffer from long term depedency/forgetting issue.
Thus, they are a feed forward model implementing auto-regressiveness.

### Training

My main confusions is usually how the decoder model makes (generates) predictions at train vs test/inference.

For training the model is trained via teacher forcing. It receives the true target as input to the decoder as if that is
what it's predicting. The target will be right shifted but it will only attend to the left because **the first 
Multi-Headed-Attention (MHA) mechanism is masked**. This is what makes the transformer -- especially the decoder -- an 
autoregressive model. This not how the encoder is implemented -- the encoder sees everything & thus is bidirectional. 
These two things together allow the Decoder to never cheat when generating outputs. In addition, this is why you 
do **not** need to loop through he sequence length during training which would be prohibitively expensive. 
It might be able to be done in cuda but people don't do this. 
Thus, transformers are rarely trained with their own predictions.

### Testing (Inference/Eval)

Since my main confusion is how the decoder generates prediction at testing.

Recall that the Decoder's first layer is a **masked** Multi-Headed-Attention (mMHA) layer. Therefore, it can only attend
to the left and doesn't cheat when generating predictions. 
At testing we do **not** have the truth e.g. we are producing a document for a user. Therefore, the model is forced to
use its own output as input auto-regressively. There is no way around it, the model has to have a loop over the length
feeding the newly predicted token -- likely until <eos> is seen or max context length is reached. 
This is ok because at inference we don't repeadely take hunderds of thousands of steps like we do at train time -- thus,
it is ok if the model seems more innefficient at eval time. The only problem is if you wrapped an eval loop that loops
through the seq length at training. However, if you want at every training step to eval a simple solution is to still 
use teacher forcing -- but keeping in mind the test/val accs might be higher than they should but it's a fine trade off.
You can do early stopping but LLM rarely need it. 

Conclusion: you need to loop through the seq length at generation/prediction time in **true** eval mode.

note: it seems there is prefixed causual masked for decoders i.e. the condition is a mask with ones, instead of 
predicting and ignoring the outputs according to the t5 video (& paper?).

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
