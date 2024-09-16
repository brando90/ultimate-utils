"""
Note: one disadvantage I hypothesize about qlora is that although it's hypothesized that qlora
fixes the errors introduced by 4 bit quant, the truth is that a perfect eval to check this is very hard. In particular,
it might only fix the errors up to the relevant fine-tuning task. Even if MMLU results are still good (due to imperfect
benchmark). The qlora model might only perform well up to some limits of what the fine-tuning data wants it to do.
Might be worth checking, asking Tim.
"""
from argparse import Namespace
import torch

from trl import SFTTrainer

from pdb import set_trace as st

from uutils.hf_uu.hf_argparse.falcon_uu_training_args import get_training_arguments_falcon7b

def train_falcon_qlora_ft(args: Namespace):
    import uutils
    from uutils.wandb_uu.sweeps_common import setup_wandb_for_train_with_hf_trainer

    # - init run, if report_to is wandb then: 1. sweep use online args merges with sweep config, else report_to is none and wandb is disabled
    config, run = setup_wandb_for_train_with_hf_trainer(args)
    uutils.pprint_any_dict(config)

    # - the get datasets todo: preprocessing, padding, streaming
    from uutils.hf_uu.data_hf.common import get_guanaco_datsets_add_splits_train_test_only
    trainset, _, testset = get_guanaco_datsets_add_splits_train_test_only()

    # qlora flacon7b
    # from uutils.hf_uu.model_tokenizer.falcon_uu_mdl_tok import get_model_tokenizer_qlora_falcon7b
    # model, tokenizer, peft_config = get_model_tokenizer_qlora_falcon7b()
    from uutils.hf_uu.model_tokenizer.falcon_uu_mdl_tok import get_model_tokenizer_qlora_falcon7b_default
    model, tokenizer, peft_config = get_model_tokenizer_qlora_falcon7b_default()
    from uutils.hf_uu.common import print_dtype_hf_model_torch
    print_dtype_hf_model_torch(model)

    # training_arguments
    # training_arguments = get_training_arguments_falcon7b()
    from uutils.hf_uu.hf_argparse.falcon_uu_training_args import get_original_training_args
    training_arguments = get_original_training_args()

    # - qlora-ft (train)
    max_seq_length = 512  # todo, get from config
    trainer = SFTTrainer(
        model=model,
        train_dataset=trainset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # todo: why this?
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()

    # Inference
    prompt = f'''What is the difference between nuclear fusion and fission?
    ###Response:'''

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(
        inputs=input_ids,
        temperature=0.7,
        max_new_tokens=512,
        top_p=0.15,
        top_k=0,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    st()

    # Finish the current run
    run.finish()

