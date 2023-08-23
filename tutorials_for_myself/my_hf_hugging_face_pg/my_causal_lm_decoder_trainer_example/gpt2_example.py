from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel, TextDataset, \
    DataCollatorForLanguageModeling
from transformers import Trainer


# Define the custom argument classes
@dataclass
class ModelArguments:
    # Holds the pre-trained model name
    model_name_or_path: Optional[str] = field(default="gpt2")


@dataclass
class DataArguments:
    # Holds the path to the training data
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class CustomTrainingArguments(TrainingArguments):
    # Holds training arguments, extends transformers.TrainingArguments
    cache_dir: Optional[str] = field(default=None)  # Directory for caching pre-trained models
    model_max_length: int = field(default=128, metadata={"help": "Maximum sequence length."})  # Maximum sequence length


def load_dataset(data_path, tokenizer):
    # Loads and tokenizes the dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_path,
        block_size=128)

    # Prepares data for Masked Language Modeling (MLM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return dataset, data_collator


def train():
    # Parse script arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path)

    # Load and tokenize dataset
    train_dataset, data_collator = load_dataset(data_args.data_path, tokenizer)

    # Initialize model
    model = GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        prediction_loss_only=True,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model()


if __name__ == "__main__":
    train()
