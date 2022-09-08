# %%
"""
Goal: use a standard off-the-shelf tokenizer (i.e. HuggingFace BPE) but
    then re-train it on the statistics of your code dataset.

refs:
- https://huggingface.co/docs/transformers/v4.21.1/en/fast_tokenizers
"""

# from transformers import AutoTokenizer, XLMRobertaTokenizer
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast, RobertaTokenizerFast

path: Path = Path('~/data/tmp/').expanduser()
path.mkdir(parents=True, exist_ok=True)

# https://huggingface.co/docs/transformers/v4.21.1/en/model_doc/roberta#transformers.RobertaTokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
assert tokenizer.is_fast
print(tokenizer)
print(tokenizer.vocab_size)

# - save tokenizer
tokenizer.save_pretrained(path / "pre-train-roberta-base")
tokenizer.save_pretrained("pre-train-roberta-base")

# - try out tokenizer
# tokenizer_checkpoint = "saved-tokenizer"
tokenizer_checkpoint = path / "pre-train-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
print(tokenizer)
print(f'{type(tokenizer)=}')

# - test tokenizer

print()
print("Sanity check: Convert a list of lists of token ids into a list of strings by calling decode.")
# - Convert a list of lists of token ids into a list of strings by calling decode.
print(f'{tokenizer.decode([0, 1, 2])=}')
print(f'{tokenizer.decode([0, 1, 2, 3, 4, 5, 6, 7, 8])=}')

print("Sanity check: Main method to tokenize and prepare for the model one or several sequence(s) "
      "or one or several pair(s) of sequences. I think this is calling __call__ or encode?")
print(tokenizer('<s><pad></s>'))
print(tokenizer('<s><pad></s><unk>. the, to and'))

print('\nSanity check: let\'s tokenize something for real:')
in_text = 'Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading' \
          ' pretrained tokenizers as well as adding tokens to the vocabulary.'
print(in_text)

print('\nEncode text in tokenizer\'s ids:')
encoding: list[int] = tokenizer(in_text)['input_ids']
print(f'{encoding=}')

print('\nSanity check: check original text and decoded text are of similar length')
print(f'{len(in_text.split())=}')
print(f'{len(encoding)=}')

print('\nSanity check: decode output ids of the tokenizer back to a string, is it similar to original input text?')
print(f'{tokenizer.decode(encoding)=}')
print(f'{in_text=}')


# print('Print str of each token.')
# print('\n'.join([tokenizer.decode(token_id) for token_id in encoding]))

# - retraining a tokenizer (i.e. get new token stats), write made up coq code or something.
def my_iterator():
    data = [
        " \n\nTheorem plus_comm: forall n m: nat, n + m = m + n.\n Proof.\n   intros n m.  induction n as [|n1 IHn1].\n   - rewrite <- plus_n_O.  simpl.  reflexivity.\n   - rewrite <- plus_n_Sm.  simpl.  rewrite -> IHn1. "
        "(fun n m : nat =>\n nat_ind (fun n0 : nat => n0 + m = m + n0)\n   (eq_ind m (fun n0 : nat => 0 + m = n0) eq_refl (m + 0) (plus_n_O m))\n   (fun (n1 : nat) (IHn1 : n1 + m = m + n1) =>\n    eq_ind (S (m + n1)) (fun n0 : nat => S n1 + m = n0)\n      (eq_ind_r (fun n0 : nat => S n0 = S (m + n1)) eq_refl IHn1) \n      (m + S n1) (plus_n_Sm m n1)) n)",

        " \n\n\nTheorem plus_assoc: forall n m p: nat, n + (m + p) = (n + m) + p.\n   intros n m p.  induction n as [| n1 IHn1].\n   - simpl.  reflexivity.\n   - simpl.  rewrite -> IHn1.  reflexivity.\n",
        "(fun n m p : nat =>\n nat_ind (fun n0 : nat => n0 + (m + p) = n0 + m + p) eq_refl\n   (fun (n1 : nat) (IHn1 : n1 + (m + p) = n1 + m + p) =>\n    eq_ind_r (fun n0 : nat => S n0 = S (n1 + m + p)) eq_refl IHn1) n)"
    ]
    for data_point in data:
        yield data_point


print()
tokenizer_checkpoint = path / "pre-train-roberta-base"
tokenizer: RobertaTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
print(f'original tokenizer vocab size: {tokenizer.vocab_size=}')
vocab_size_new_guess: int = 500
vocab_size: int = tokenizer.vocab_size + vocab_size_new_guess  # todo how do you know this value if you've not ran the tokenizer yet?
print(f'{vocab_size=}')
# - this re-train your tokenizer from scratch
new_tokenizer: PreTrainedTokenizerFast = tokenizer.train_new_from_iterator(my_iterator(), vocab_size=vocab_size)
print(f'original tokenizer vocab size: {tokenizer.vocab_size=}')
print(f'new vocab size: {new_tokenizer.vocab_size=}')
# assert new_tokenizer.vocab_size > tokenizer.vocab_size, f'new tokenizer vocab size should be at least as large as original.'

tokenizer_checkpoint = path / "new-pre-train-roberta-base"
new_tokenizer.save_pretrained(tokenizer_checkpoint)

# - test new tokenizer
tokenizer = new_tokenizer
print('\nSanity check: let\'s tokenize something for real:')
in_text = '(fun n m p : nat =>\n nat_ind (fun n0 : nat => n0 + (m + p) = n0 + m + p) eq_refl\n   (fun (n1 : nat) (IHn1 : n1 + (m + p) = n1 + m + p) =>\n    eq_ind_r (fun n0 : nat => S n0 = S (n1 + m + p)) eq_refl IHn1) n)'
print(in_text)

print('\nEncode text in tokenizer\'s ids:')
encoding: list[int] = tokenizer(in_text)['input_ids']
print(f'{encoding=}')

print('\nSanity check: check original text and decoded text are of similar length')
print(f'{len(in_text.split())=}')
print(f'{len(encoding)=}')

print('\nSanity check: decode output ids of the tokenizer back to a string, is it similar to original input text?')
print(f'{tokenizer.decode(encoding)=}')
print(f'{in_text=}')

print('Success! \n\a')
