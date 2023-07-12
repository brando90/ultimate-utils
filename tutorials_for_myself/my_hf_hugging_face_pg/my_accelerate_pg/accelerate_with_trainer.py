"""
colab: https://colab.research.google.com/drive/1hcIwNjETTpbWjlKGkvjyDVzB1SvRR4il?authuser=0#scrollTo=rEdmwDr9oODh

https://chat.openai.com/share/fe0f8984-1444-4f0f-9ce4-c0fd681903d3
"""
# %%
# !pip install accelerate
# !pip install datasets
# !pip install transformers

# %%
"""
accelerate launch --config_file {path/to/config/my_config_file.yaml} {script_name.py} {--arg1} {--arg2} ...
"""
from accelerate import Accelerator
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TrainingArguments, Trainer

# Initialize accelerator
accelerator = Accelerator()

# Specify dataset
dataset = load_dataset('imdb')

# Specify tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(accelerator.device)


# Tokenize and format dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=accelerator.num_processes,
    remove_columns=["text"]
)

# Training configuration
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    # num_train_epochs=3,
    max_steps=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=False,  # Set to True for mixed precision training (FP16)
    fp16_full_eval=False,  # Set to True for mixed precision evaluation (FP16)
    dataloader_num_workers=accelerator.num_processes,  # Use multiple processes for data loading
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# Train model
trainer.train()
