import os
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from pathlib import Path

# Load Hugging Face token from file
with open(Path("~/keys/writing_2_brando_amazob_aws_2024_summer.txt").expanduser(), "r") as file:
    hf_token = file.read().strip()

# Set the Hugging Face token as an environment variable
os.environ["HF_TOKEN"] = hf_token

# Login using the token
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

# Load model and tokenizer
pretrained_model_name_or_path = "openai-community/gpt2"
if 'gpt2' in pretrained_model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'{tokenizer.pad_token=}')
    print(f'{tokenizer.eos_token=}\n{tokenizer.eos_token_id=}')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    block_size: int = tokenizer.model_max_length
    print(f'{block_size=}')

# Define training arguments with memory optimization tricks
training_args = TrainingArguments(
    output_dir="~/tmp/results",  # Output directory for saving model checkpoints
    per_device_train_batch_size=1,  # Training batch size per device
    per_device_eval_batch_size=1,  # Evaluation batch size per device
    max_steps=2,  # Total number of training steps
    logging_dir='~/tmp/logs',  # Directory for storing logs
    logging_steps=10,  # Frequency of logging steps
    gradient_accumulation_steps=1,  # Accumulate gradients to simulate a larger batch size
    save_steps=500,  # Save checkpoint every 500 steps
    save_total_limit=3,  # Only keep the last 3 checkpoints
    evaluation_strategy="steps",  # Evaluate model at specified steps
    eval_steps=100,  # Evaluate every 100 steps
    gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
    optim="paged_adamw_32bit",  # Optimizer choice with memory optimization
    learning_rate=1e-5,  # Learning rate for training
    warmup_ratio=0.01,  # Warmup ratio for learning rate schedule
    weight_decay=0.01,  # Weight decay for regularization
    lr_scheduler_type='cosine',  # Learning rate scheduler type
    report_to="none",  # Disable reporting to external tracking tools
    # bf16=torch.cuda.is_bf16_supported(),  # Use BF16 if supported by the hardware
    half_precision_backend="auto",  # Automatically select the best backend for mixed precision
    # dataloader_num_workers=4,  # TODO Number of subprocesses for data loading
    # dataloader_pin_memory=True,  # TODO periphery, Pin memory in data loaders for faster transfer to GPU
    # skip_memory_metrics=True,  # Skip memory metrics to save memory
    # dataloader_prefetch_factor=2,  # TODO periphery, Number of batches to prefetch
    # torchdynamo="nvfuser",  # TODO periphery, Use NVFuser backend for optimized torch operations
    full_determinism=True,  # TODO periphery, Ensure reproducibility
)

# Load the C4 dataset in streaming mode
train_dataset = load_dataset("c4", "en", split="train", streaming=True)
eval_dataset = load_dataset("c4", "en", split="validation", streaming=True)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

# Start training
trainer.train()

# Save the model and tokenizer
trainer.save_model(output_dir="~/tmp/results")
tokenizer.save_pretrained(output_dir="~/tmp/results")
