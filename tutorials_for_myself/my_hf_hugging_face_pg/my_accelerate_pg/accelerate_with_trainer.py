"""
"""
# %%
# !pip install accelerate
# !pip install datasets
# !pip install transformers

# %%
"""
accelerate launch --config_file {path/to/config/my_config_file.yaml} {script_name.py} {--arg1} {--arg2} ...
accelerate obj not needed due to seamless integration with hf trainer: https://github.com/huggingface/accelerate/issues/144
"""
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TrainingArguments, Trainer

# Specify dataset
dataset = load_dataset('imdb')

# Specify tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Tokenize and format dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
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
