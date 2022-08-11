# HuggingFace (HF) notes

## Coupling Tokenizer & new Models is a good idea

- since models word embedding layer is depedent of arbitrary token index & size of tokenizer, it makes sense to me to:
  1. save/checkpoint & load model & tokenizer together always (check hf's checkpointing default)
  2. when getting a brand new model, I suggest to either
     a. pass the tokenizer being used (e.g. if it's re-trained, then re-shape model's token emebd to match it.)
- **main take away is to make sure when you have a new model, to have it match the tokenizer your going to use**


- Let hf checkpointing take care of this?

## Fine-tuning **both** model & tokenizer with new data

- TODO: see Q1

## Questions:

- Q1: how to extend a tokenizer with custom new tokens & the model from a pre-trained model and fine-tune everything.

```
For now, this behavior is kept to avoid breaking backwards compatibility when padding/encoding with `truncation is True`.
- Be aware that you SHOULD NOT rely on t5-small automatically truncating your input to 512 when padding/encoding.
- If you want to encode/pad to sequences longer than 512 you can either instantiate this tokenizer with `model_max_length` or pass `max_length` when encoding/padding.
- To avoid this warning, please instantiate this tokenizer with `model_max_length` set to your preferred value.
  warnings.warn(
```