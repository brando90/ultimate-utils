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
import torch


def get_model_tokenizer_qlora_falcon7b(model_name: str = "ybelkada/falcon-7b-sharded-bf16",
                                       config: wand.Config, # todo
                                       lora_alpha=16, # todo
                                       lora_dropout=0.1, # todo
                                       lora_r=64, # todo
                                       ) -> tuple:
    """
    Load the Falcon 7B model, quantize it in 4bit and attach LoRA adapters on it.

    bf16 = 1S, 7Exp, 8Mantissa
    ref:
        - https://colab.research.google.com/drive/1DOi8MFv4SWN9NImVornZ7t6BgmLoPQO-#scrollTo=AjB0WAqFSzlD
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

    # model_id = "tiiuae/falcon-7b"
    # model_name: str = "ybelkada/falcon-7b-sharded-bf16"

    # - get bnb config for int8 model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # todo: looks weird, check later
    )

    # - get falcon int8 model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model.config.use_cache = False  # todo: why?

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    from peft import LoraConfig

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

