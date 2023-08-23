# @dataclass
# class ModelArguments:
#     model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
#
#
# @dataclass
# class DataArguments:
#     data_path: str = field(default=None, metadata={"help": "Path to the training data."})
#
#
# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="adamw_torch")
#     model_max_length: int = field(
#         default=512,
#         metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
#     )
#
#
# def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
#     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
#     return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
#
#
# def train():
#     parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#
# #%%
# # url: https://chat.openai.com/share/b61ad56e-05cc-4c61-93fb-0ed0221f6b13
# import os
# import yaml
# import wandb
# from dataclasses import dataclass, field
# from typing import Optional
# from transformers import HfArgumentParser, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel, TextDataset, \
#     DataCollatorForLanguageModeling
# from transformers import Trainer
#
#
# # Define the custom argument classes
# @dataclass
# class ModelArguments:
#     model_name_or_path: Optional[str] = field(default="gpt2")
#
#
# @dataclass
# class DataArguments:
#     data_path: str = field(default=None, metadata={"help": "Path to the training data."})
#
#
# @dataclass
# class CustomTrainingArguments(TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     model_max_length: int = field(default=128, metadata={"help": "Maximum sequence length."})
#
#
# def load_dataset(data_path, tokenizer):
#     dataset = TextDataset(
#         tokenizer=tokenizer,
#         file_path=data_path,
#         block_size=128)
#
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer, mlm=False,
#     )
#     return dataset, data_collator
#
#
# def train():
#     # Automatically starts a new run with the hyperparameters chosen by the sweep
#     run = wandb.init()
#
#     # The hyperparameters for this run are automatically added to wandb.config
#     sweep_config = run.config
#     from argparse import Namespace
#     sweep_config: Namespace = Namespace(**sweep_config)
#     args: list[str] = [item for pair in [[f'--{k}', str(v)] for k, v in sweep_config.items()] for item in pair]
#
#     # Parse script arguments
#     parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args)
#
#     # # The hyperparameters for this run are automatically added to wandb.config
#     # sweep_config = dict(run.config)
#     # # Convert sweep_config to a list of strings in the format that argparse would recognize
#     # args = []
#     # for key, value in sweep_config.items():
#     #     args.append(f'--{key}')
#     #     args.append(str(value))
#     # # Parse script arguments
#     # parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
#     # model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args)
#
#     # Update training arguments from sweep
#     training_args.learning_rate = sweep_config.learning_rate
#     training_args.num_train_epochs = sweep_config.num_train_epochs
#
#     # Initialize tokenizer
#     tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path)
#
#     # Load and tokenize dataset
#     train_dataset, data_collator = load_dataset(data_args.data_path, tokenizer)
#
#     # Initialize model
#     model = GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path)
#
#     # Initialize trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         data_collator=data_collator,
#         prediction_loss_only=True,
#     )
#
#     # Train the model
#     trainer.train()
#
#     # Save the model
#     trainer.save_model()
#
#
# if __name__ == "__main__":
#     # Load sweep config
#     with open('sweep_config.yaml') as file:
#         sweep_config = yaml.load(file, Loader=yaml.FullLoader)
#
#     # Initialize a new sweep
#     # Arguments:
#     # 1. Project name
#     # 2. Sweep configuration
#     sweep_id = wandb.sweep(sweep_config, project="gpt2-sweep")
#
#     # Run the sweep
#     # Arguments:
#     # 1. Sweep ID
#     # 2. Function to call
#     # 3. Number of runs
#     wandb.agent(sweep_id, function=train, count=5)
