"""

"""
from pathlib import Path
from typing import Optional

import yaml
from transformers import HfArgumentParser


# # -- falcon_peft.py ref: https://gist.github.com/pacman100/1731b41f7a90a87b457e8c5415ff1c14
# @dataclass
# class ScriptArguments:
#     """
#     These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
#     """
#
#     local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
#
#     per_device_train_batch_size: Optional[int] = field(default=4)
#     per_device_eval_batch_size: Optional[int] = field(default=1)
#     gradient_accumulation_steps: Optional[int] = field(default=4)
#     learning_rate: Optional[float] = field(default=2e-4)
#     max_grad_norm: Optional[float] = field(default=0.3)
#     weight_decay: Optional[int] = field(default=0.001)
#     lora_alpha: Optional[int] = field(default=16)
#     lora_dropout: Optional[float] = field(default=0.1)
#     lora_r: Optional[int] = field(default=64)
#     max_seq_length: Optional[int] = field(default=512)
#     model_name: Optional[str] = field(
#         default="tiiuae/falcon-7b",
#         metadata={
#             "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
#         },
#     )
#     dataset_name: Optional[str] = field(
#         default="timdettmers/openassistant-guanaco",
#         metadata={"help": "The preference dataset to use."},
#     )
#     use_4bit: Optional[bool] = field(
#         default=True,
#         metadata={"help": "Activate 4bit precision base model loading"},
#     )
#     use_nested_quant: Optional[bool] = field(
#         default=False,
#         metadata={"help": "Activate nested quantization for 4bit base models"},
#     )
#     bnb_4bit_compute_dtype: Optional[str] = field(
#         default="float16",
#         metadata={"help": "Compute dtype for 4bit base models"},
#     )
#     bnb_4bit_quant_type: Optional[str] = field(
#         default="nf4",
#         metadata={"help": "Quantization type fp4 or nf4"},
#     )
#     num_train_epochs: Optional[int] = field(
#         default=1,
#         metadata={"help": "The number of training epochs for the reward model."},
#     )
#     fp16: Optional[bool] = field(
#         default=False,
#         metadata={"help": "Enables fp16 training."},
#     )
#     bf16: Optional[bool] = field(
#         default=False,
#         metadata={"help": "Enables bf16 training."},
#     )
#     packing: Optional[bool] = field(
#         default=False,
#         metadata={"help": "Use packing dataset creating."},
#     )
#     gradient_checkpointing: Optional[bool] = field(
#         default=True,
#         metadata={"help": "Enables gradient checkpointing."},
#     )
#     optim: Optional[str] = field(
#         default="paged_adamw_32bit",
#         metadata={"help": "The optimizer to use."},
#     )
#     lr_scheduler_type: str = field(
#         default="constant",
#         metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
#     )
#     max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
#     warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
#     group_by_length: bool = field(
#         default=True,
#         metadata={
#             "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
#         },
#     )
#     save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
#     logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
#
#
# parser = HfArgumentParser(ScriptArguments)
# script_args = parser.parse_args_into_dataclasses()[0]


def _legacy_get_args_for_run_from_cmd_args_or_sweep(parser: HfArgumentParser,
                                                    path2sweep_config: Optional[str] = None,
                                                    ) -> list[tuple]:
    """
    Warning: decided against this.

    Parses the arguments from the command line
    note:
        - if you pass the general parser then it will decide how to organize the tuple of args for you already.

    ref:
        - if ever return to this approach: https://stackoverflow.com/questions/76585219/what-is-the-official-way-to-run-a-wandb-sweep-with-hugging-face-hf-transformer
    """
    # 1. parse all the arguments from the command line
    # parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments, GeneralArguments))
    # _, _, _, general_args = parser.parse_args_into_dataclasses()  # default args is to parse sys.argv
    # 2. if the wandb_config option is on, then overwrite run cmd line configuration in favor of the sweep_config.
    if path2sweep_config:  # None => False => not getting wandb_config
        # overwrite run configuration with the wandb_config configuration (get config and create new args)
        config_path = Path(path2sweep_config).expanduser()
        with open(config_path, 'r') as file:
            sweep_config = dict(yaml.safe_load(file))
        sweep_args: list[str] = [item for pair in [[f'--{k}', str(v)] for k, v in sweep_config.items()] for item in
                                 pair]
        args: tuple = parser.parse_args_into_dataclasses(args=sweep_args)
        # model_args, data_args, training_args, general_args = parser.parse_args_into_dataclasses(args=sweep_args)
        # args: tuple = (model_args, data_args, training_args, general_args)  # decided against named obj to simplify code
        # 3. execute run from sweep
        # Initialize the sweep in Python which create it on your project/eneity in wandb platform and get the sweep_id.
        # sweep_id = wandb.sweep(sweep_config, entity=sweep_config['entity'], project=sweep_config['project'])
        # # Finally, once the sweep_id is acquired, execute the sweep using the desired number of agents in python.
        # train = lambda : train(args)  # pkg train with args i.e., when you call train() it will all train(args).
        # wandb.agent(sweep_id, function=train, count=general_args.count)
        # # print(f"Sweep URL: https://wandb.ai/{sweep_config['entity']}/{sweep_config['project']}/sweeps/{sweep_id}")
        # wandb.get_sweep_url()
    else:
        # use the args from the command line
        model_args, data_args, training_args, general_args = parser.parse_args_into_dataclasses()
        # 3. execute run
        args: tuple = (model_args, data_args, training_args, general_args)  # decided against named obj to simplify code
        # train(args)
    return args
