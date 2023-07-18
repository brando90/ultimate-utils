import sys
from typing import Optional

import torch
from torch import bfloat16, float16
from peft import LoraConfig
from transformers import PreTrainedTokenizerFast

from transformers.modeling_utils import PreTrainedModel

from pdb import set_trace as st

def get_model_tokenizer_camel_5b_hf(pretrained_model_name_or_path: str = "Writer/camel-5b-hf",
                                    use_cache: bool = False,  # False saves gpu mem ow keeps more in mem for speed
                                    verbose: bool = True,
                                    ) -> tuple[PreTrainedModel, PreTrainedTokenizerFast, Optional[LoraConfig]]:
    """ ref: https://gpt-index.readthedocs.io/en/latest/examples/customization/llms/SimpleIndexDemo-Huggingface_camel.html """
    from uutils.hf_uu.common import hf_dist_print
    # from uutils import get_filtered_local_params
    # get_filtered_local_params(locals(), verbose=verbose, var_name_in_front='training_arguments') if verbose else None

    # -- Loading the model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # -- Loading the model (type will be RWForCausalLM)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
    model.config.use_cache = use_cache  # False saves gpu mem ow True keeps more mdl stuff in mem for speed gains
    hf_dist_print(f'{type(model)=}')

    # -- Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    hf_dist_print(f'{type(tokenizer)=}')
    hf_dist_print(f'{tokenizer.model_max_length=}')
    # tokenizer.pad_token = tokenizer.eos_token
    # add_brand_new_pad_token_to_tokenizer_falcon(tokenizer, model)
    return model, tokenizer, None