
def example_merge_wandb_config_with_args_using_hf_parser():
    """
    todo main thing to test is:
    - if the names in wandb_config and my dataclasses have to match. Hopefully wandb_config can have less and the defualts are used.
    """
    import wandb
    from transformers import HfArgumentParser, TrainingArguments
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class ModelArguments:
        model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

    @dataclass
    class DataArguments:
        data_path: str = field(default=None, metadata={"help": "Path to the training data."})

    @dataclass
    class TrainingArguments(TrainingArguments):
        cache_dir: Optional[str] = field(default=None)

    run = wandb.init()
    wandb.get_sweep_url()
    sweep_config = run.config
    # might need to change a little bit to respect the wandb_config structure
    args: list[str] = [item for pair in [[f'--{k}', str(v)] for k, v in sweep_config.items()] for item in pair]
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args)
    # make sure the 3 or X args have the fields from the wandb_config