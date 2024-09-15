import os
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from typing import Dict, Tuple, Optional
from pathlib import Path
import evaluate

# from utils import eval_hf, get_ai4m_v0, get_data_set_args, load_dataset_block_size
from utils import eval_hf

# from utils import load_model_block_size


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray],
                    path: str = 'accuracy',
                    ) -> Dict[str, float]:
    """
    Compute the accuracy of the model.

    Args:
    eval_pred: A tuple containing the model predictions and labels.

    Returns:
    A dictionary with the accuracy score.
    
    TODO: document properly what accuracy is. Is it tfa, ara, exact string match, avg acc (wrt length etc.) ref: https://huggingface.co/spaces/evaluate-metric/accuracy
    """
    metric = evaluate.load(path=path)   # load metric from file or hf
    predictions, references = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=references)

def preprocess_function_proofnet_simple(examples: Dict[str, list], tokenizer: GPT2Tokenizer, max_length: int = 512) -> Dict[str, torch.Tensor]:
    """
    Preprocess the input data for the proofnet dataset.

    Args:
    examples: The examples to preprocess.
    tokenizer: The tokenizer for encoding the texts.

    Returns:
    The processed model inputs.
    """
    inputs = [f"{examples['nl_statement'][i]}{tokenizer.eos_token}{examples['formal_statement'][i]}" for i in range(len(examples['nl_statement']))]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = model_inputs.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

def setup_and_train_proofnet(
        # pretrained_model_name_or_path: str = "gpt2", 
        # pretrained_model_name_or_path: str = "openai-community/gpt2-xl", 
        pretrained_model_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B", 
        path: str = "hoskinson-center/proofnet",
        output_dir_train: str = '~/tmp/proofnet/train',
        output_dir_val: Optional[str] = None,  # we are training on the val set so no val set
        output_dir_test: str = '~/tmp/proofnet/test',
        path_to_save_model: Optional[str] = None,  # suggested path: '~/tmp/proofnet/model' then expanduser in py code
        num_train_epochs: int = 5,
        per_device_train_batch_size: Optional[int] = 2,
        per_device_eval_batch_size: Optional[int] = 2,
        save_total_limit: Optional[int] = None,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0, 
        optim='paged_adamw_32bit',
        gradient_accumulation_steps = 2, # see: based on alpaca https://github.com/tatsu-lab/stanford_alpaca, allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
        gradient_checkpointing: Optional[bool] = False,
        # lr_scheduler_type='cosine',  # TODO: https://discord.com/channels/879548962464493619/1227708244697284724/1227708244697284724
        # warmup_ratio=0.01,   # TODO: https://discord.com/channels/879548962464493619/1227708244697284724/1227708244697284724
        report_to: str = 'none',  # recommended values 'wandb' or `none`
        ) -> None:
    """
    Set up the environment, preprocess the dataset, and train the model.

    export CUDA_VISIBLE_DEVICES=7

    Args:
    tokenizer_name: The name of the tokenizer.
    model_name: The name of the model.
    dataset_path: The path to the dataset.
    """
    # Clear CUDA cache to free up memory
    torch.cuda.empty_cache()

    # Load tokenizer and model
    if pretrained_model_name_or_path == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, max_length=1024)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f'{tokenizer.pad_token=}')
        print(f'{tokenizer.eos_token=}\n{tokenizer.eos_token_id=}')
        model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        block_size: int = tokenizer.model_max_length
        print(f'{block_size=}')
    elif pretrained_model_name_or_path == "openai-community/gpt2-xl":
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, max_length=1024)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f'{tokenizer.pad_token=}')
        print(f'{tokenizer.eos_token=}\n{tokenizer.eos_token_id=}')
        model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        block_size: int = tokenizer.model_max_length
        print(f'{block_size=}')
    else:
        raise ValueError(f"Model {pretrained_model_name_or_path} not supported.")
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    # Load the dataset
    dataset_val = load_dataset(path, split='validation')
    dataset_test = load_dataset(path, split='test')

    # Preprocess the dataset
    if path == "hoskinson-center/proofnet":
        preprocess_function = preprocess_function_proofnet_simple
        # note: text field is usually more common!
        val_dataset = dataset_val.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])
        test_dataset = dataset_test.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])

    # Training arguments
    output_dir_train: Path = Path(output_dir_train).expanduser()
    output_dir_train.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir_train,
        evaluation_strategy='no',  # "no"`: No evaluation is done during training. no can be good to avoid memory issues.
        gradient_accumulation_steps=gradient_accumulation_steps,  # based on alpaca https://github.com/tatsu-lab/stanford_alpaca, allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
        gradient_checkpointing = gradient_checkpointing,  # TODO depending on hardware set to true?
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        num_train_epochs=num_train_epochs,
        max_grad_norm=max_grad_norm,
        optim=optim,
        logging_first_step=True,
        logging_strategy='epoch',
        # lr_scheduler_type=lr_scheduler_type  # TODO: https://discord.com/channels/879548962464493619/1227708244697284724/1227708244697284724
        # warmup_ratio=warmup_ratio,
        report_to = report_to,  # options I recommend: 'none', 'wandb'
        fp16=False,  # never ever set to True
        bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8,  # if >= 8 ==> brain float 16 available or set to True if you always want fp32
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=val_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # Train the model
    trainer.train()

    # Evaluate the model
    if output_dir_test is not None:
        output_dir_test: Path = Path(output_dir_test).expanduser()
        output_dir_test.mkdir(parents=True, exist_ok=True)
        eval_args = TrainingArguments(output_dir=output_dir_test, fp16=False, bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8, report_to=report_to)
        trainer = Trainer(model=model, args=eval_args, train_dataset=None, eval_dataset=test_dataset)
        # results: dict[str, float] = trainer.evaluate(test_dataset)
        results: dict[str, float] = eval_hf(trainer, name='', path=path, split='test')
        print(f'{path=} split=test {results=}')

    # Save the trained model
    if path_to_save_model is not None:
        model.save_pretrained(path_to_save_model)

def main() -> None:
    """
    Main function to execute the model training and evaluation.
    """
    setup_and_train_proofnet()

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print(f"Time taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
