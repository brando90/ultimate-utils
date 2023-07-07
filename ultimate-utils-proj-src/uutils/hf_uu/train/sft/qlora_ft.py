from argparse import Namespace
from trl import SFTTrainer

import uutils
from uutils.wandb_uu.sweeps_common import setup_wandb_for_train_with_hf_trainer


def train_falcon(args: Namespace):
    # - init run, if report_to is wandb then: 1. sweep use online args merges with sweep config, else report_to is none and wandb is disabled
    config, run = setup_wandb_for_train_with_hf_trainer(args)
    print(f'{config=}')
    uutils.pprint_any_dict(config)

    # - the get datasets todo: preprocessing, padding, streaming
    from uutils.hf_uu.data_hf.common import get_guanaco_datsets_add_splits_train_test_only
    trainset, _, testset = get_guanaco_datsets_add_splits_train_test_only()

    # qlora flacon7b
    from uutils.hf_uu.model_tokenizer.falcon_uu import get_model_tokenizer_qlora_falcon7b
    model, tokenizer, peft_config, training_arguments = get_model_tokenizer_qlora_falcon7b()

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

    # Finish the current run
    run.finish()
