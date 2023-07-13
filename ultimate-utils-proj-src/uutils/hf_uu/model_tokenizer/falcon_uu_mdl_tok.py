"""
sfttrainer (likely using peft) best practices:
https://huggingface.co/docs/trl/main/en/sft_trainer#best-practices

Best practices

Pay attention to the following best practices when training a model with that trainer:

- SFTTrainer always pads by default the sequences to the max_seq_length argument of the SFTTrainer. If none is passed, the trainer will retrieve that value from the tokenizer. Some tokenizers do not provide default value, so there is a check to retrieve the minimum between 2048 and that value. Make sure to check it before training.
- For training adapters in 8bit, you might need to tweak the arguments of the prepare_model_for_int8_training method from PEFT, hence we advise users to use prepare_in_int8_kwargs field, or create the PeftModel outside the SFTTrainer and pass it.
- For a more memory-efficient training using adapters, you can load the base model in 8bit, for that simply add load_in_8bit argument when creating the SFTTrainer, or create a base model in 8bit outside the trainer and pass it.
- If you create a model outside the trainer, make sure to not pass to the trainer any additional keyword arguments that are relative to from_pretrained() method.

todo: why trust_remote_code? I want more details.
"""
import sys
from typing import Optional

import torch
from torch import bfloat16, float16
from peft import LoraConfig
from transformers import PreTrainedTokenizerFast

from transformers.modeling_utils import PreTrainedModel

from pdb import set_trace as st


def get_any_bnb_quant_config(quantize_mode: str):
    from transformers import BitsAndBytesConfig
    if quantize_mode == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quantize_mode == "8bit":
            bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif quantize_mode == "16bit":
        bnb_config = BitsAndBytesConfig(
        )
    return bnb_config



def add_brand_new_pad_token_to_tokenizer_falcon(tokenizer: PreTrainedTokenizerFast,
                                                model: PreTrainedModel,
                                                pad_token: str = '[PAD]',
                                                ):
    """ Add pad token to tokenizer and model. This is needed for the model to work with the trainer.
    note: this function cannot be made general due to not impossibility of extracting the word embedding layer for any
    hf model due to specific names of layers.
    """
    tokenizer.add_special_tokens({'pad_token': pad_token})  # I think this is fine if during the training pad is ignored

    model.resize_token_embeddings(len(tokenizer))  # todo: I think this is fine if during the training pad is ignored
    model.transformer.word_embeddings.padding_idx = len(tokenizer) - 1
    model.config.max_new_tokens = len(tokenizer)
    model.config.pad_token_id = len(tokenizer) - 1
    model.config.min_length = 1


def get_model_tokenizer_qlora_falcon7b(
        # -- mode args
        # model_id = "tiiuae/falcon-7b"
        pretrained_model_name_or_path: str = "ybelkada/falcon-7b-sharded-bf16",
        use_cache: bool = True,  # this is here to save gpu vram. Likely only needed when using 40b.
        # -- lora args
        lora_alpha=16,  # todo
        lora_dropout=0.1,  # todo, evidence drop out really help? google, crfm, gpt4
        lora_r=64,  # todo
        # if you want to overwrite comp dtype pass it explicitly, for now defualt to bfloat if available else float16
        # bnb_4bit_compute_dtype=torch.float16,  # changed it from Guanaco hf
        # bnb_4bit_compute_dtype=torch.bfloat16,  # changed it from Guanaco hf
        bnb_4bit_compute_dtype=None,  # changed it from Guanaco hf

        # -- training args
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        # paging so that the sudden mem gpu spikes don't cause the run to shut down
        # (I think usually caused by too long seqs)
        # todo: why 32 bit opt?
        # todo: paged nadamw opt?
        optim="paged_adamw_32bit",
        save_steps=10,
        logging_steps=10,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        max_steps=500,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        # -- quant. args (not recommended to be changed unless you know what your doing?)
        load_in_4bit=True,  # load (usually huge) base model in 4 bits
        bnb_4bit_quant_type="nf4",  # normal float 4 for the (large) base models qlora
) -> tuple:
    """
    Load the Falcon 7B model, quantize it in 4bit and attach LoRA adapters on it.

    bf16 = 1S, 7Exp, 8Mantissa
    hypothesis: 7b trained due to 6.7 emergence rumour, I still don't think emergence is real.
    Notes:
        - ft a model is very specific to the model, tokenizer and training scheme. Thus we return
            - model, tokenizer, ft config (peft config), training args

    ref:
        - https://colab.research.google.com/drive/1DOi8MFv4SWN9NImVornZ7t6BgmLoPQO-#scrollTo=AjB0WAqFSzlD
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

    # try to use bfloat16 if no comp. dtype is given
    if bnb_4bit_compute_dtype is None:  # only if not given try the bfloat16 default or float16 default
        from uutils.torch_uu import bfloat16_avail
        bnb_4bit_compute_dtype = bfloat16 if bfloat16_avail() else float16

    # - Get bnb config for bit-4 base model (bnb lib for using 4bit qlora quantization techniques by tim dettmers)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,  # load (usually huge) base model in 4 bits
        bnb_4bit_quant_type=bnb_4bit_quant_type,  # normal float 4 for the (usually huge) base model
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,  # if you can, during computation use bf16
    )

    # - Get falcon 4bit model
    # todo, where is this being saved & how to download quicker
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True  # allows to execute custom code you download from the uploaded model code you are using
    )
    print(f'{type(model)=}')
    print(f'{model=}')
    # this is here to save gpu vram. Likely only needed when using 40b or when oom issues happen ref: https://stackoverflow.com/questions/76633335/why-does-hugging-face-falcon-model-use-mode-config-use-cache-false-why-wouldn
    model.config.use_cache = use_cache  # likely only needed when oom issues happen
    print(f'{type(model)=}')

    # - Get falcon tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                              trust_remote_code=True)  # execs code downloaded from hf hub
    tokenizer.pad_token = tokenizer.eos_token  # todo: this is wrong since it pads out eos and learns to not ouptut eos after fine-tuning ref: https://stackoverflow.com/questions/76633368/why-does-the-falcon-qlora-tutorial-code-use-eos-token-as-pad-token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # I think this is fine if during the training pad is ignored
    # tokenizer.add_special_tokens({'pad_token': '<|pad|>'})  # I think this is fine if during the training pad is ignored
    # tokenizer.pad_token = '<|pad|>'

    # - Modify model
    # add pad token embed
    # model.resize_token_embeddings(len(tokenizer))  # todo: I think this is fine if during the training pad is ignored
    # model.transformer.word_embeddings.padding_idx = len(tokenizer) - 1
    # model.config.max_new_tokens = len(tokenizer)
    # model.config.pad_token_id = len(tokenizer) - 1
    # model.config.min_length = 1
    # print(f'{model=}')
    # print(f'{type(tokenizer)=}')
    # print(f'{tokenizer.pad_token=}')
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) todo

    # - Get falcon lora config
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        # model card for falcon tiiuae/falcon-7b: https://huggingface.co/tiiuae/falcon-7b/blob/main/modelling_RW.py
        # does seem to include all trainable params as done by qlora on their own paper
        target_modules=[
            # word_embeddings, # todo, check in original pdf qlora, else try to fine-tune it too? oh might introduce params...?
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
            # "lm_head"
        ]
    )
    print(f'{type(peft_config)=}')

    # todo: print the num params of the lora = D1*r + D2*r and num of bytes by prec. (bytes) * num params, code peft_uu
    return model, tokenizer, peft_config


def get_model_tokenizer_qlora_falcon7b_default() -> tuple:
    """
    directly from https://colab.research.google.com/drive/1DOi8MFv4SWN9NImVornZ7t6BgmLoPQO-#scrollTo=AjB0WAqFSzlD
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

    # Loading the model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

    model_name = "ybelkada/falcon-7b-sharded-bf16"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model.config.use_cache = False

    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Lora
    from peft import LoraConfig

    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ]
    )
    return model, tokenizer, peft_config


def get_model_tokenizer_fp32_falcon(pretrained_model_name_or_path: str = "tiiuae/falcon-7b",
                                    use_cache: bool = False,  # False saves gpu mem ow keeps more in mem for speed
                                    verbose: bool = True,
                                    ) -> tuple[PreTrainedModel, PreTrainedTokenizerFast, Optional[LoraConfig]]:
    """ Get Falcon model and Tokenizer.

    :param pretrained_model_name_or_path: "tiiuae/falcon-7b" or "tiiuae/falcon-1b" see: https://huggingface.co/tiiuae
    """
    from uutils import get_filtered_local_params
    get_filtered_local_params(locals(), verbose=verbose, var_name_in_front='training_arguments') if verbose else None

    from uutils.hf_uu.common import hf_dist_print
    # Loading the model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

    # Loading the model (type will be RWForCausalLM)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True
    )
    model.config.use_cache = use_cache  # False saves gpu mem ow True keeps more mdl stuff in mem for speed gains
    hf_dist_print(f'{type(model)=}')

    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
    hf_dist_print(f'{type(tokenizer)=}')
    hf_dist_print(f'{tokenizer.model_max_length=}')
    # tokenizer.pad_token = tokenizer.eos_token
    add_brand_new_pad_token_to_tokenizer_falcon(tokenizer, model)
    return model, tokenizer, None


# -- tests

def example_test_model_already_has_pad_token():
    """
    if it already has pad token, it likely has a small prob, so we are done.

    compare it's norm with other tokens to verify this is true.

python ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/model_tokenizer/falcon_uu_mdl_tok.py
    """
    # - the get datasets todo: preprocessing, padding, streaming
    from uutils.hf_uu.data_hf.common import get_guanaco_datsets_add_splits_train_test_only
    trainset, _, testset = get_guanaco_datsets_add_splits_train_test_only()

    # qlora flacon7b
    from uutils.hf_uu.model_tokenizer.falcon_uu_mdl_tok import get_model_tokenizer_qlora_falcon7b
    model, tokenizer, peft_config = get_model_tokenizer_qlora_falcon7b()
    model: PreTrainedModel = model.to('cuda')
    print(f'{model=}')
    sent = 'Dogs are great because they are '
    print()

    # print to see if pad tokens are present and if it ignores the tokens at the end
    encoded_input = tokenizer(sent, padding='max_length', max_length=10, return_tensors='pt')
    print(f'{encoded_input=}')

    # Print all special tokens
    print('\n---- start Print all special tokens')
    for token_name, token in tokenizer.special_tokens_map.items():
        print(f"{token_name}: {token}")
    print('\n---- end Print all special tokens')

    # Get the ID for the '[PAD]' token
    try:
        pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
    except KeyError:
        raise ValueError("Token [PAD] is not present in the tokenizer vocabulary.")

    # Index into the model's embedding table
    try:
        print(f'{model.get_input_embeddings().weight.size()=}')
        pad_embedding = model.get_input_embeddings().weight[pad_token_id]
    except IndexError:
        raise ValueError(f"Token ID {pad_token_id} is not present in the model's embedding matrix.")

    print(f'{pad_embedding=}')
    print('Success!\n')

    # check it generates something sensible
    # tokenizer.decode(model.generate(**tokenizer(sent, return_tensors='pt'), do_sample=True)[0])
    input_ids, attention_mask = encoded_input['input_ids'].to('cuda'), encoded_input['attention_mask'].to('cuda')
    predicted_tokens_ids_options = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=True)
    predicted_tokens_ids = predicted_tokens_ids_options[0]
    predicted_sent = tokenizer.decode(predicted_tokens_ids)
    print(f'original sentence: {sent=}')
    print(f'predicted sentence: {predicted_sent=}')
    print('Success2!')


if __name__ == '__main__':
    import time

    start_time = time.time()
    example_test_model_already_has_pad_token()
    print(f"The main function executed in {time.time() - start_time} seconds.\a")
