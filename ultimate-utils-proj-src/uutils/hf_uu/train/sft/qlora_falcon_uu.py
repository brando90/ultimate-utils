from transformers import HfArgumentParser

from uutils.hf_uu.hf_argparse.falcon_uu import ModelArguments, DataArguments, TrainingArguments
from uutils.wandb_uu.sweeps_common import exec_run_for_wandb_sweep


def train(args: tuple):
    run = wandb.init()
    print(f'{wandb.get_sweep_url()=}')

    sweep_config = run.config
    # might need to change a little bit to respect the wandb_config structure
    args: list[str] = [item for pair in [[f'--{k}', str(v)] for k, v in sweep_config.items()] for item in pair]
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args)

    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    #     model_max_length=training_args.model_max_length,
    #     padding_side="right",  # Ensures properly masking out the source tokens.
    #     use_fast=training_args.use_fast_tokenizer,
    # )
    # tokenizer.padding = training_args.padding
    # - accelerate ref: https://chat.openai.com/share/1014a48a-d714-472f-9285-d6baa419fe6b
    # accelerator = Accelerator()
    # model, optimizer, train_dataset, test_dataset = accelerator.prepare(model, optimizer, train_dataset, test_dataset)

    data_module: dict = data_utils.make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    # Tokenizer is only supplied so that it gets saved; this makes loading easier.
    # SFTTrainer best practices: https://huggingface.co/docs/trl/main/en/sft_trainer#best-practices
    # trainer = Trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     **data_module,
    # )
    from trl import SFTTrainer

    max_seq_length = 512

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # todo put if statement, if using lora do this, or always do it basically actually?
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # logger.warning("hooray! training finished successfully! now on to model saving.", main_process_only=True)  # todo: seems complicated for now, ignore. We can use it or put our own detection if we are main process, I already had code from uutils that checked this and then printed.
    print("hooray! training finished successfully! now on to model saving.")
    trainer.save_state()
    # model.save_pretrained("output_dir")
    # model.push_to_hub("my_awesome_peft_model") also works
    # common.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)  # todo: seems like a deepspeed optimization, will keep it simple for now.
    # logger.warning("hooray again! model saving worked.", main_process_only=True)
    print("hooray again! model saving worked.")


def main_falcon():
    """
    Simply executes a run using the wandb config.
    """
    # 1. parse all the arguments from the command line
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()  # default args is to parse sys.argv

    # 2. prepare train func
    args: tuple = (model_args, data_args, training_args)
    train: callable = lambda: train(args)  # train() now calls train(args) in wandb.agent.

    # 3. execute train run from sweep
    path2sweep_config: str = training_args.path2sweep_config
    exec_run_for_wandb_sweep(path2sweep_config, train)


if __name__ == '__main__':
    import time

    start_time = time.time()
    main_falcon()
    print(f"The main function executed in {time.time() - start_time} seconds.\a")
