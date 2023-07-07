from argparse import Namespace
import torch

import uutils
from uutils.wandb_uu.sweeps_common import exec_run_for_wandb_sweep, setup_wandb_for_train_with_hf_trainer


def train_falcon(args: Namespace):
    # - init run, if report_to is wandb then: 1. sweep use online args merges with sweep config, else report_to is none and wandb is disabled
    config, run = setup_wandb_for_train_with_hf_trainer(args)
    print(f'{config=}')
    uutils.pprint_any_dict(config)

    # - the get datasets todo: preprocessing, padding, streaming
    from uutils.hf_uu.data_hf.common import get_guanaco_datsets_add_splits_train_test_only
    trainset, _, testset = get_guanaco_datsets_add_splits_train_test_only()

    # qlora flacon7b
    model, tokenizer, peft_config = get_model_tokenizer_qlora_falcon7b()

    # - qlora-ft (train)
    from trl import SFTTrainer
    max_seq_length = 512
    trainer = SFTTrainer(
        model=model,
        train_dataset=trainset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()

    # Finish the current run
    run.finish()

def main_falcon():
    """
python -m pdb -c continue /Users/brandomiranda/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/main_falcon_uu.py --report_to none
python -m pdb -c continue /Users/brandomiranda/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/main_falcon_uu.py --report_to wandb
    """
    from uutils.hf_uu.hf_argparse.common import get_simple_args

    # - get most basic hf args args
    args: Namespace = get_simple_args()  # just report_to, path2sweep_config, path2debug_seep
    print(args)

    # - run train
    report_to = args.report_to
    if report_to == "none":
        train: callable = train_falcon
        train(args)
    elif report_to == "wandb":
        path2sweep_config = args.path2sweep_config
        train = lambda: train_falcon(args)
        exec_run_for_wandb_sweep(path2sweep_config, train)
    else:
        raise ValueError(f'Invaid hf report_to option: {report_to=}.')

if __name__ == '__main__':
    import time

    start_time = time.time()
    main_falcon()
    print(f"The main function executed in {time.time() - start_time} seconds.\a")