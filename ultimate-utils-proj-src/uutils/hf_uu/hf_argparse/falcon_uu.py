from dataclasses import dataclass, field
from typing import Literal, Optional

import transformers

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
    path2sweep_config: str = field(default=None, metadata={"help": "Path to the wandb sweep config .yaml file."})
    report_to: str = field(default="none", metadata={"help": "Report to wandb or none."})
