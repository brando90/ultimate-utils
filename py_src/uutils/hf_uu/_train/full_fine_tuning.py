from argparse import Namespace

from pdb import set_trace as st


def train_falcon_7b_fp32(args: Namespace):
    import torch
    import uutils
    from uutils.wandb_uu.sweeps_common import setup_wandb_for_train_with_hf_trainer  # todo: check wandd.config is None?
    from uutils.hf_uu.common import assert_model_dtype_is, print_dtype_hf_model_torch
    from uutils.hf_uu.common import estimate_memory_used_by_loaded_model_no_data
    from uutils.hf_uu.common import hf_dist_print

    # -- Check GPU memory
    from uutils.hf_uu.common import print_gpu_memory_usage
    print_gpu_memory_usage()  # just to see total available memory but doesn't reflect what pytorch does idk why whatevs

    # -- Init wand run. if report_to is wandb then: 1. sweep use online args merges with sweep config, else report_to is none and wandb is disabled
    config, run = setup_wandb_for_train_with_hf_trainer(args)
    uutils.pprint_any_dict(config, var_name_in_front='config')

    # -- Get DataSet Splits todo: preprocessing, padding, streaming
    hf_dist_print('---- Get DataSet Splits')
    from uutils.hf_uu.data_hf.common import get_guanaco_datsets_add_splits_train_test_only
    trainset, _, testset = get_guanaco_datsets_add_splits_train_test_only()

    # -- Get Model: flacon7b
    hf_dist_print('---- Get Model')
    from uutils.hf_uu.model_tokenizer.falcon_uu_mdl_tok import get_model_tokenizer_fp32_falcon
    model, tokenizer, peft_config = get_model_tokenizer_fp32_falcon()
    hf_dist_print(f'{model=}')
    hf_dist_print(f'{tokenizer=}')
    # estimate_memory_used_by_loaded_model_no_data(model, 'cuda', verbose=True)  # only for single gpu
    print_dtype_hf_model_torch(model)
    assert_model_dtype_is(torch.float32, model, num_layers=1)

    # -- Get Training Arguments
    hf_dist_print('---- Get Training Arguments')
    # training_arguments = get_training_arguments_falcon7b()
    from uutils.hf_uu.hf_argparse.falcon_uu_training_args import get_training_args_falcon_7b_fp32
    training_arguments = get_training_args_falcon_7b_fp32(report_to=args.report_to)
    # print(f'{training_arguments=}')

    # - full-fine tune (train) -- using SFTrainer since has same API as Trainer but also accepts peft params
    # The max_seq_length is the max length for input seq (tr & eval) truncating longer sequences and padding shorter ones, it will be fixed and helps predicibility OOM errs. https://chat.openai.com/share/f4e48c26-b729-42d2-8a3c-600b3dc587e8
    max_seq_length = 512  # todo, get from config, use above comment
    # tokenizer.model_max_length = max_seq_length   # perhaps more general...actually SFTrainer same interface as Trainer so passing it in SFTrainer is fine. Just use SFTrainer instead of Trainer.
    hf_dist_print(f'{max_seq_length=}')
    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=trainset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    assert_model_dtype_is(torch.float32, model, num_layers=1)
    trainer.train()

    # Finish the current run
    run.finish()
