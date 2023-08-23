# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import transformers
from transformers import Trainer

from alpaca_farm import common, constants, data_utils, logging, utils

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Name to a huggingface native pretrained model or path to a model on disk."}
    )


@dataclass
class DataArguments:
    dataset_path: str = field(
        default="tatsu-lab/alpaca_farm",
        metadata={
            "help": "Path to the dataset. Either points to a location on Hugging Face hub or a local folder. "
            "If the path points to a local folder, the folder must be structured properly "
            "(see documentation for datasets.load_dataset)."
        },
    )
    dataset_name: Optional[str] = field(
        default="alpaca_instructions",
        metadata={"help": "Name of the dataset to load -- the argument `name` passed to `datasets.load_dataset`."},
    )
    train_splits: List[str] = field(
        default_factory=lambda: ["sft"],
        metadata={"help": "Splits to use for training. This must not be an empty list."},
    )
    eval_splits: Optional[List[str]] = field(
        default_factory=lambda: ["val"],
        metadata={
            "help": "Splits to use for evaluation. "
            "If None, empty, or the splits are not found in the dataset, no evaluation is performed."
        },
    )
    prompt_dict_path: str = field(
        default=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
        metadata={"help": "Path to the dictionary for the prompt to format examples."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pad_token: str = field(default=constants.DEFAULT_PAD_TOKEN)
    cache_dir: str = field(default=constants.DEFAULT_CACHE_DIR)
    wandb_project: str = field(default=constants.WANDB_PROJECT)
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