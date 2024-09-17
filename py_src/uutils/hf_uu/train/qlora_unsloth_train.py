"""
ref: https://chatgpt.com/c/66e9c6cb-cfb0-8001-8c77-bec486e00a6b
"""
import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import fire

def formatting_informalization(example, tokenizer, EOS_TOKEN):
    # Must add EOS_TOKEN, otherwise your generation will go on forever!
    informalization_prompt = """
    Below is a natural language explanation of a Theorem from the Lean4 Mathlib library.
    {}
    """
    text = informalization_prompt.format(example["informalization"]) + EOS_TOKEN
    return {"text": text}

def main(
        model_name: str = "unsloth/llama-3-8b",
        output_dir: str = "outputs",
        dataset_name: str = "AI4M/leandojo-informalized",
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 1,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        max_seq_length: int = 8192,
        warmup_steps: int = 5,
        use_4bit: bool = False,
        r_lora: int = 16,
        logging_steps: int = 1,
        seed: int = 3407,
        push_to_hub: Optional[bool] = False,
        model_save_name: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detection for Float16/BFloat16
        load_in_4bit=use_4bit,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=r_lora,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )

    # Load dataset and preprocess
    leandojo_informalized_dataset = load_dataset(dataset_name, split="train")
    EOS_TOKEN = tokenizer.eos_token
    dataset = leandojo_informalized_dataset.map(
        lambda example: formatting_informalization(example, tokenizer, EOS_TOKEN), batched=False
    )

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=logging_steps,
        optim="adamw_8bit",
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        seed=seed,
        output_dir=output_dir,
    )

    # Initialize the Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Train the model
    print(f"Starting training for {num_train_epochs} epoch(s)...")
    trainer_stats = trainer.train()

    # Show memory usage
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{used_memory} GB of memory reserved.")
    print(f"{used_memory_for_lora} GB used for LoRA.")
    print(f"{used_percentage}% of GPU memory used.")
    print(f"{lora_percentage}% used for training LoRA.")

    # Save the trained model
    if push_to_hub and model_save_name and hf_token:
        model.push_to_hub_merged(
            model_save_name,
            tokenizer=tokenizer,
            save_method="merged_16bit",
            token=hf_token,
        )
    elif model_save_name:
        model.save_pretrained_merged(model_save_name, tokenizer, save_method="merged_16bit")

if __name__ == "__main__":
    import time
    start_time = time.time()
    fire.Fire(main)
    print(f"Time taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
