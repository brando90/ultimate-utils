from typing import Union

import torch
from datasets import Dataset, concatenate_datasets
from torch import nn
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoModel, PreTrainedTokenizer


def add_special_all_special_tokens(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
    """
        special_tokens_dict = {"cls_token": "<CLS>"}

        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print("We have added", num_added_toks, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))

        assert tokenizer.cls_token == "<CLS>"

    """
    original_len: int = len(tokenizer)
    num_added_toks: dict = {}
    if tokenizer.bos_token is None:
        num_added_toks['bos_token'] = "<bos>"
    if tokenizer.bos_token is None:
        num_added_toks['cls_token'] = "<cls>"
    if tokenizer.bos_token is None:
        num_added_toks['sep_token'] = "<s>"
    if tokenizer.bos_token is None:
        num_added_toks['mask_token'] = "<mask>"
    # num_added_toks = {"bos_token": "<bos>", "cls_token": "<cls>", "sep_token": "<s>", "mask_token": "<mask>"}
    # special_tokens_dict = {'additional_special_tokens': new_special_tokens + tokenizer.all_special_tokens}
    num_new_tokens: int = tokenizer.add_special_tokens(num_added_toks)
    assert tokenizer.bos_token == "<bos>"
    assert tokenizer.cls_token == "<cls>"
    assert tokenizer.sep_token == "<s>"
    assert tokenizer.mask_token == "<mask>"
    err_msg = f"Error, not equal: {len(tokenizer)=}, {original_len + num_new_tokens=}"
    assert len(tokenizer) == original_len + num_new_tokens, err_msg


def get_model_given_a_tokenizer(tokenizer: PreTrainedTokenizerFast,
                                model: nn.Module,
                                add_my_special_tokens: bool = False,
                                ):
    # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained('t5-small')
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    if add_my_special_tokens:
        add_special_all_special_tokens(tokenizer)
    # model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    assert isinstance(model, torch.nn.Module)
    model.resize_token_embeddings(len(tokenizer))
    assert model.get_input_embeddings().weight.size(0) == len(tokenizer)


def get_tokenizer_trained_from_scratch_and_acompanying_model(datasets: Union[list[Dataset], Dataset],
                                                             path2save_tokenizer,
                                                             pretrained_model_name_or_path,
                                                             add_my_special_tokens: bool = False,
                                                             ) \
        -> tuple[PreTrainedTokenizerFast, nn.Module]:
    """

    want:
        - train the tokenizer form scratch on my data set (forget finetuning the tokenizer & model for now)
        - make sure you return a model that works for the new tokenizer
    """
    from uutils.torch_uu.data_uu.hf_uu_tokenizer import re_train_tokenizer_from
    if isinstance(datasets, list):
        dataset: Dataset = concatenate_datasets(datasets)
    else:
        dataset: Dataset = datasets

    tokenizer: PreTrainedTokenizerFast = re_train_tokenizer_from(dataset, path2save_tokenizer=path2save_tokenizer)
    # print(len(tokenizer))
    original_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    # print(len(original_tokenizer))
    assert len(tokenizer) != len(original_tokenizer), 'If new tokenizer and old are exactly the same size recheck.'  # very unlucky if they are same size
    if add_my_special_tokens:
        add_special_all_special_tokens(tokenizer)
    # model = AutoModel.from_pretrained(pretrained_model_name_or_path)
    from transformers import AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path)
    assert isinstance(model, torch.nn.Module)
    model.resize_token_embeddings(len(tokenizer))
    assert model.get_input_embeddings().weight.size(0) == len(tokenizer)
    return tokenizer, model
