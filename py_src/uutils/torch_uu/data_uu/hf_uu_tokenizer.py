import os
from typing import Union

from datasets import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer


# - retraining a tokenizer (i.e. get new token stats), write made up coq code or something.

def re_train_tokenizer_from(dataset: Dataset,
                            vocab_size_new_guess: int = 500,
                            pretrained_model_name_or_path: Union[str, os.PathLike] = "t5-small",
                            path2save_tokenizer=None,
                            verbose: bool = False,
                            ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """
    """
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path)
    if verbose:
        print(f'{isinstance(tokenizer, PreTrainedTokenizer)=}')
        print(f'{isinstance(tokenizer, PreTrainedTokenizerFast)=}')
        print(f'{type(tokenizer)=}')
        print(tokenizer)
        print(f'{tokenizer.vocab_size=}')

    vocab_size: int = tokenizer.vocab_size + vocab_size_new_guess  # todo how do you know this value if you've not ran the tokenizer yet?
    if verbose:
        print(f'{vocab_size=}')
    # - this re-train your tokenizer from scratch
    new_tokenizer: PreTrainedTokenizerFast = tokenizer.train_new_from_iterator(iter(dataset), vocab_size=vocab_size)
    if verbose:
        print(f'original tokenizer vocab size: {tokenizer.vocab_size=}')
        print(f'new vocab size: {new_tokenizer.vocab_size=}')
    # assert new_tokenizer.vocab_size > tokenizer.vocab_size, f'new tokenizer vocab size should be at least as large as original.'
    if path2save_tokenizer:
        new_tokenizer.save_pretrained(path2save_tokenizer)
    # assert len(new_tokenizer) != len(tokenizer)  # very unlucky if they are same size
    return new_tokenizer


def my_iterator():
    data = [
        " \n\nTheorem plus_comm: forall n m: nat, n + m = m + n.\n Proof.\n   intros n m.  induction n as [|n1 IHn1].\n   - rewrite <- plus_n_O.  simpl.  reflexivity.\n   - rewrite <- plus_n_Sm.  simpl.  rewrite -> IHn1. "
        "(fun n m : nat =>\n nat_ind (fun n0 : nat => n0 + m = m + n0)\n   (eq_ind m (fun n0 : nat => 0 + m = n0) eq_refl (m + 0) (plus_n_O m))\n   (fun (n1 : nat) (IHn1 : n1 + m = m + n1) =>\n    eq_ind (S (m + n1)) (fun n0 : nat => S n1 + m = n0)\n      (eq_ind_r (fun n0 : nat => S n0 = S (m + n1)) eq_refl IHn1) \n      (m + S n1) (plus_n_Sm m n1)) n)",

        " \n\n\nTheorem plus_assoc: forall n m p: nat, n + (m + p) = (n + m) + p.\n   intros n m p.  induction n as [| n1 IHn1].\n   - simpl.  reflexivity.\n   - simpl.  rewrite -> IHn1.  reflexivity.\n",
        "(fun n m p : nat =>\n nat_ind (fun n0 : nat => n0 + (m + p) = n0 + m + p) eq_refl\n   (fun (n1 : nat) (IHn1 : n1 + (m + p) = n1 + m + p) =>\n    eq_ind_r (fun n0 : nat => S n0 = S (n1 + m + p)) eq_refl IHn1) n)"
    ]
    for data_point in data:
        yield data_point


def get_toy_custom_trained_t5_tokenizer(
        vocab_size_new_guess: int = 500
) -> PreTrainedTokenizerFast:
    """ """
    # tokenizer_checkpoint = path / "pre-train-roberta-base"
    # tokenizer: RobertaTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("t5-small")
    print(f'{isinstance(tokenizer, PreTrainedTokenizer)=}')
    print(f'{isinstance(tokenizer, PreTrainedTokenizerFast)=}')
    print(f'{type(tokenizer)=}')
    print(tokenizer)
    print(tokenizer.vocab_size)

    vocab_size: int = tokenizer.vocab_size + vocab_size_new_guess  # todo how do you know this value if you've not ran the tokenizer yet?
    print(f'{vocab_size=}')
    # - this re-train your tokenizer from scratch
    new_tokenizer: PreTrainedTokenizerFast = tokenizer.train_new_from_iterator(my_iterator(), vocab_size=vocab_size)
    print(f'original tokenizer vocab size: {tokenizer.vocab_size=}')
    print(f'new vocab size: {new_tokenizer.vocab_size=}')
    # assert new_tokenizer.vocab_size > tokenizer.vocab_size, f'new tokenizer vocab size should be at least as large as original.'
    return tokenizer


# - tests

def nice_preprocess_of_tokenizer_example(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
    batch_sentences = [
        "But what about second breakfast?",
        "Don't think he knows about second breakfast, Pip.",
        "What about elevensies?",
    ]
    encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
    print(encoded_input)
    """
    {'input_ids': tensor([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
                          [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
                          [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]]), 
     'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
    """

def does_t5_have_sep_token():
    """

    https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
    https://discuss.huggingface.co/t/issue-with-finetuning-a-seq-to-seq-model/1680/29
    https://stackoverflow.com/questions/73322462/how-to-add-all-standard-special-tokens-to-my-hugging-face-tokenizer-and-model
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained('t5-small')
    # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained('t5-base')
    # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained('t5-large')
    # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained('t5-11b')
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    print(tokenizer)
    print(f'{len(tokenizer)=}')

    print()
    print(f'{tokenizer.sep_token=}')
    print(f'{tokenizer.eos_token=}')
    print(f'{tokenizer.all_special_tokens=}')
    print()

    special_tokens_dict = {'additional_special_tokens': ['<bos>', '<cls>', '<s>'] + tokenizer.all_special_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # tokenizer.add_tokens([f"_{n}" for n in range(1, 100)], special_tokens=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    assert isinstance(model, torch.nn.Module)
    model.resize_token_embeddings(len(tokenizer))
    # tokenizer.save_pretrained('pathToExtendedTokenizer/')
    # tokenizer = T5Tokenizer.from_pretrained("sandbox/t5_models/pretrained/tokenizer/")

    # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained('t5-small')
    new_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    assert isinstance(model, torch.nn.Module)
    model.resize_token_embeddings(len(new_tokenizer))

    print()
    print(f'{tokenizer.bos_token=}')
    print(f'{tokenizer.cls_token=}')
    print(f'{tokenizer.sep_token=}')
    print(f'{tokenizer.mask_token=}')
    print(f'{tokenizer.eos_token=}')
    print(f'{tokenizer.unk_token=}')
    print(f'{tokenizer.bos_token_id=}')
    print(f'{tokenizer.cls_token_id=}')
    print(f'{tokenizer.sep_token_id=}')
    print(f'{tokenizer.mask_token_id=}')
    print(f'{tokenizer.eos_token_id=}')
    print(f'{tokenizer.unk_token_id=}')
    print(f'{tokenizer.all_special_tokens=}')
    print()



if __name__ == '__main__':
    does_t5_have_sep_token()
    print('Done\a')

