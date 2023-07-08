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

import torch
from peft import LoraConfig

from transformers.modeling_utils import PreTrainedModel

from pdb import set_trace as st


def test_bfloat16_int4(compute_dtype: torch.dtype,
                       use_4bit,
                       ):
    """
python -c "import torch; print(torch.cuda.get_device_capability());"
    todo: check other code test_bfloat16() do we need use_4bit?
    """
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bfloat16")
            print("=" * 80)


def get_model_tokenizer_qlora_falcon7b(
        # -- mode args
        # model_id = "tiiuae/falcon-7b"
        pretrained_model_name_or_path: str = "ybelkada/falcon-7b-sharded-bf16",
        use_cache: bool = True,
        # -- lora args
        lora_alpha=16,  # todo
        lora_dropout=0.1,  # todo, evidence drop out really help? google, crfm, gpt4
        lora_r=64,  # todo
        bnb_4bit_compute_dtype=torch.float16,  # changed it from Guanaco hf

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
    model.config.use_cache = use_cache
    print(f'{type(model)=}')

    # - Get falcon tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                              trust_remote_code=True)  # execs code downloaded from hf hub
    # tokenizer.pad_token = tokenizer.eos_token  # todo: why? https://stackoverflow.com/questions/76633368/why-does-the-falcon-qlora-tutorial-code-use-eos-token-as-pad-token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # I think this is fine if during the training pad is ignored
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})  # I think this is fine if during the training pad is ignored
    model.resize_token_embeddings(len(tokenizer))  # todo: I think this is fine if during the training pad is ignored
    model.transformer.word_embeddings.padding_idx = len(tokenizer) - 1
    print(f'{model=}')
    print(f'{type(tokenizer)=}')
    print(f'{tokenizer.pad_token=}')
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
            # word_embeddings,
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
            # "lm_head"
        ]
    )
    print(f'{type(peft_config)=}')

    # todo: print the num params of the lora = D1*r + D2*r and num of bytes by prec. (bytes) * num params
    return model, tokenizer, peft_config


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
    model: PreTrainedModel = model
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
    input_ids, attention_mask = encoded_input['input_ids'], encoded_input['attention_mask']
    predicted_tokens_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=True)[0]
    predicted_sent = tokenizer.decode(predicted_tokens_ids)
    print(f'original sentence: {sent=}')
    print(f'predicted sentence: {predicted_sent=}')
    print('Success2!')


if __name__ == '__main__':
    import time

    start_time = time.time()
    example_test_model_already_has_pad_token()
    print(f"The main function executed in {time.time() - start_time} seconds.\a")
