from argparse import Namespace

from pdb import set_trace as st

def train_falcon_7b_32fp_28gb_mem(args: Namespace):
    import torch
    import uutils
    from uutils.wandb_uu.sweeps_common import setup_wandb_for_train_with_hf_trainer  # todo: check wandd.config is None?
    from uutils.hf_uu.common import assert_model_dtype_is

    # -- Check GPU memory
    from uutils.hf_uu.common import print_gpu_memory_usage
    print_gpu_memory_usage()  # just to see total available memory but doesn't reflect what pytorch does idk why whatevs

    # -- Init wand run. if report_to is wandb then: 1. sweep use online args merges with sweep config, else report_to is none and wandb is disabled
    config, run = setup_wandb_for_train_with_hf_trainer(args)
    uutils.pprint_any_dict(config)

    # -- Get DataSet Splits todo: preprocessing, padding, streaming
    print('---- Get DataSet Splits')
    from uutils.hf_uu.data_hf.common import get_guanaco_datsets_add_splits_train_test_only
    trainset, _, testset = get_guanaco_datsets_add_splits_train_test_only()

    # -- Get Model: flacon7b
    print('---- Get Model')
    from uutils.hf_uu.model_tokenizer.falcon_uu_mdl_tok import get_model_tokenizer_fp32_falcon
    model, tokenizer, peft_config = get_model_tokenizer_fp32_falcon()
    from uutils.hf_uu.common import estimate_memory_used_by_loaded_model_no_data
    estimate_memory_used_by_loaded_model_no_data(model, 'cuda', verbose=True)
    st()
    from uutils.hf_uu.common import print_dtype_hf_model_torch
    print_dtype_hf_model_torch(model)
    assert_model_dtype_is(torch.float32, model, num_layers=1)

    # training_arguments
    # training_arguments = get_training_arguments_falcon7b()
    from uutils.hf_uu.hf_argparse.falcon_uu_training_args import get_training_args_falcon_7b_32fp_28gb_mem
    training_arguments = get_training_args_falcon_7b_32fp_28gb_mem(report_to=args.report_to)

    # - full-fine tune (train) -- using SFTrainer since has same API as Trainer but also accepts peft params
    max_seq_length = 512  # todo, get from config
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
