import torch
from datasets import Dataset, concatenate_datasets
from torch import nn
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast, T5ForConditionalGeneration, AutoTokenizer


def get_t5_from_path_to_custom_tokenizer(tokenizer: PreTrainedTokenizerFast,
                                         model: T5ForConditionalGeneration,
                                         add_my_special_tokens: bool = False,
                                         ):
    # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained('t5-small')
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    if add_my_special_tokens:
        special_tokens_dict = {'additional_special_tokens': ['<bos>', '<cls>', '<s>'] + tokenizer.all_special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
    # model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    assert isinstance(model, torch.nn.Module)
    model.resize_token_embeddings(len(tokenizer))
    # new_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_ckpt)
    model.resize_token_embeddings(len(tokenizer))


def get_t5_by_finetuning_tokenizer_and_model_from_pretrained_tokenizer_and_model():
    pass


def make_pretrained_model_comaptible_from_pretrained_tokenizer():
    pass


def get_tokenizer_trained_from_scratch_and_companying_model(datasets: list[Dataset],
                                                            path2save_tokenizer,
                                                            pretrained_model_name_or_path: str = 't5-small',
                                                            add_my_special_tokens: bool = False,
                                                            ) \
        -> tuple[PreTrainedTokenizerFast, T5ForConditionalGeneration]:
    """

    want:
        - train the tokenizer form scratch on my data set (forget finetuning the tokenizer & model for now)
        - make sure you return a model that works for the new tokenizer
    """
    from uutils.torch_uu.data_uu.hf_uu_tokenizer import re_train_tokenizer_from
    dataset: Dataset = concatenate_datasets(datasets)

    tokenizer: PreTrainedTokenizerFast = re_train_tokenizer_from(dataset, path2save_tokenizer=path2save_tokenizer)
    print(len(tokenizer))
    original_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    print(len(original_tokenizer))
    assert len(tokenizer) != len(original_tokenizer)  # very unlucky if they are same size
    if add_my_special_tokens:
        special_tokens_dict = {'additional_special_tokens': ['<bos>', '<cls>', '<s>'] + tokenizer.all_special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
    model: T5ForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path)
    assert isinstance(model, torch.nn.Module)
    model.resize_token_embeddings(len(tokenizer))
    assert model.get_input_embeddings().weight.size(0) == len(tokenizer)
    return tokenizer, model
