"""
Common code to execute hf runs
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import transformers
import wandb
from conda.common.serialize import yaml
from transformers import HfArgumentParser

from uutils.hf_uu.hf_argparse.common import _legacy_get_args_for_run_from_cmd_args_or_sweep

# TODO
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),
# }

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pad_token: str = field(default=DEFAULT_PAD_TOKEN)
    # cache_dir: str = field(default=DEFAULT_CACHE_DIR)
    # wandb_project: str = field(default=WANDB_PROJECT)  # TODO
    flash_attn: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded to this length (and possibly truncated)."
                    "Enforcing a consistent max length ensures memory usage is constant and predictable."
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
                    "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    initialize_model_on_cpu: bool = field(
        default=False,
        metadata={
            "help": "Whether to initialize the model on CPU. "
                    "If True, models on all processes will be first initialized on CPU; this is RAM-costly but faster."
        },
    )
    resume_from_checkpoint: bool = field(default=False, metadata={"help": "If True, loads from last check point."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Use fast tokenizer if True. "
                    "Fast LLaMA tokenizer forces protobuf downgrade to 3.20.3. "
                    "Use fast tokenizer only if you can live with that."
        },
    )

def _legacy_main_sweep_or_no_sweep(train: callable):
    """
    Warning: decided against this because:
    - when executing runs with wandb without sweeps the user likely wants to specify a group. So this likely means we need
    to add a group option to the GeneralArguments. It would still work if the GeneralArguments had a group options when
    running in this mode, then the user specifies it in the arguments. But then in the mergning would we need to remove
    that options? To avoid this type of spaggati code, I decided against combining the sweep vs non-sweep code.

    Old: Simply execs a run either from a wand sweep file or from the command line arguments.
    Ignore the wandb sweep and only run the args from cmd if unfamilar with sweeps or sweeps code confuses you.
    Sweeps challenge is simply making usre to load confing and overwritting the args properly

    ref:
        - if ever return to this approach: https://stackoverflow.com/questions/76585219/what-is-the-official-way-to-run-a-wandb-sweep-with-hugging-face-hf-transformer
    """
    # 1. parse all the arguments from the command line
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    _, _, training_args = parser.parse_args_into_dataclasses()  # default args is to parse sys.argv
    path2sweep_config: str = training_args.path2sweep_config
    # 2. if the wandb_config option is on, then overwrite run cmd line configuration in favor of the sweep_config.
    args: tuple = _legacy_get_args_for_run_from_cmd_args_or_sweep(parser, path2sweep_config)
    train = lambda: train(args)  # pkg train with args i.e., when you call train() it will all train(args).
    if path2sweep_config:  # None => False => not getting wandb_config
        from uutils.wandb_uu.sweeps_common import exec_run_for_wandb_sweep
        exec_run_for_wandb_sweep(args, train)
    else:
        train()
    # if path2sweep_config:  # None => False => not getting wandb_config
    #     # overwrite run configuration with the wandb_config configuration (get config and create new args)
    #     config_path = Path(training_args.path2sweep_config).expanduser()
    #     with open(config_path, 'r') as file:
    #         sweep_config = dict(yaml.safe_load(file))
    #     sweep_args: list[str] = [item for pair in [[f'--{k}', str(v)] for k, v in sweep_config.items()] for item in
    #                              pair]
    #     model_args, data_args, training_args, general_args = parser.parse_args_into_dataclasses(args=sweep_args)
    #     args: tuple = (model_args, data_args, training_args, general_args)  # decided against named obj to simplify code
    #     # 3. execute run from sweep
    #     # Initialize the sweep in Python which create it on your project/eneity in wandb platform and get the sweep_id.
    #     sweep_id = wandb.sweep(sweep_config, entity=sweep_config['entity'], project=sweep_config['project'])
    #     # Finally, once the sweep_id is acquired, execute the sweep using the desired number of agents in python.
    #     train = lambda: train(args)  # pkg train with args i.e., when you call train() it will all train(args).
    #     wandb.agent(sweep_id, function=train, count=general_args.count)
    #     # print(f"Sweep URL: https://wandb.ai/{sweep_config['entity']}/{sweep_config['project']}/sweeps/{sweep_id}")
    #     wandb.get_sweep_url()
    # else:
    #     # use the args from the command line
    #     parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    #     model_args, data_args, training_args, general_args = parser.parse_args_into_dataclasses()
    #     # 3. execute run
    #     args: tuple = (model_args, data_args, training_args, general_args)  # decided against named obj to simplify code
    #     train(args)