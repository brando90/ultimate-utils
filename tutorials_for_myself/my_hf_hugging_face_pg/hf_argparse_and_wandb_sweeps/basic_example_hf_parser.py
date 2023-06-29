"""
python your_script.py --model_name_or_path "facebook/opt-125m" --data_path /path/to/training_data --cache_dir /path/to/cache --optim adamw_torch --model_max_length 512

ref: https://chat.openai.com/share/76e6bf57-3d25-4d4a-b771-1d50d0afc030
"""
import argparse
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import TrainingArguments
from transformers.utils import HFArgumentParser

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class MyTrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

# Create the HFArgumentParser instance
parser = HFArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))

# Pass the command as a string to parse_args_into_dataclasses
args = parser.parse_args_into_dataclasses(["--model_name_or_path", "facebook/opt-125m",
                                           "--data_path", "/path/to/training_data",
                                           "--cache_dir", "/path/to/cache",
                                           "--optim", "adamw_torch",
                                           "--model_max_length", "512"])

# Access the parsed arguments
model_args, data_args, training_args = args

# Use the parsed arguments as needed
print(model_args)
print(data_args)
print(training_args)
