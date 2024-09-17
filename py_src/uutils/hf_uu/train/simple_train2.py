"""

Refs:
    - https://claude.ai/chat/ad5c9e18-beb4-48fb-9f43-a2ba463ce158
    - https://chatgpt.com/c/349f2c8a-949e-444d-ae3c-8ca60ba77831
"""
import fire
from datetime import date
import glob
import os
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_metric
from typing import Dict, Tuple, Optional
from pathlib import Path
import evaluate

from utils import eval_hf
from utils import raw_ds_2_lm_ds_mask_eos_pad_toks

from pdb import set_trace as st

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

def main(
        # pretrained_model_name_or_path: str = "gpt2", 
        # pretrained_model_name_or_path: str = "openai-community/gpt2-xl", 
        # pretrained_model_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B", # note: if you get RoPE error upgrade your transformers pip install --upgrade transformers lib, https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/discussions/15
        pretrained_model_name_or_path: str = "internlm/internlm2-1_8b", # note: if you get RoPE error upgrade your transformers pip install --upgrade transformers lib, https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/discussions/15
        path: str = "hoskinson-center/proofnet",
        output_dir_train: str = '~/data/runs/run_{date}/train',
        output_dir_val: Optional[str] = None,  # we are training on the val set so no val set
        output_dir_test: str = '~/data/runs/run_{date}/test',
        path_to_save_model: Optional[str] = None,  # suggested path: '~/tmp/proofnet/model' then expanduser in py code
        num_train_epochs: int = 3,
        per_device_train_batch_size: Optional[int] = 1,
        per_device_eval_batch_size: Optional[int] = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0, 
        lr_scheduler_type = 'cosine', # https://discord.com/channels/879548962464493619/1227708244697284724/1227708244697284724
        warmup_ratio=0.01,  # copying alpaca for now, number of steps for a linear warmup,  https://discord.com/channels/879548962464493619/1227708244697284724/1227708244697284724
        optim='paged_adamw_32bit',
        gradient_accumulation_steps = 4, # Allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
        gradient_checkpointing: Optional[bool] = False, # Careful with segmentation Faults https://stackoverflow.com/questions/78841125/how-to-fix-segmentation-fault-when-training-gpt-2-model-using-hugging-face-trans
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
    print(f'->{pretrained_model_name_or_path=}')
    if pretrained_model_name_or_path == "gpt2":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32 
        # tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, max_length=1024) # TODO really, we need to put max_length here?
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f'{tokenizer.pad_token=}')
        print(f'{tokenizer.eos_token=}\n{tokenizer.eos_token_id=}')
        model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype)
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f'{device=} {torch_dtype=} Model device: {next(model.parameters()).device}')
        # max_length: int = tokenizer.model_max_length
        max_length: int = 2
        print(f'{tokenizer.model_max_length=}')
        print(f'-->{max_length=}')
    elif pretrained_model_name_or_path == "openai-community/gpt2-xl":
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, max_length=1024)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f'{tokenizer.pad_token=}')
        print(f'{tokenizer.eos_token=}\n{tokenizer.eos_token_id=}')
        model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        max_length: int = 128
        # max_length: int = 256
        # max_length: int = 512
        # max_length: int = tokenizer.model_max_length # 1024
        print(f'-->{max_length=}') 
    elif pretrained_model_name_or_path == "meta-llama/Meta-Llama-3.1-8B":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32 
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype)
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right", use_auth_token=True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right")
        print(f'{tokenizer.pad_token=} {tokenizer.eos_token_id=}')
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token
        print(f'{tokenizer.pad_token=} {tokenizer.eos_token_id=}')
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # get context length for setting max length for training
        if hasattr(model.config, "context_length"):
            # SEEMS IT IS NOT IN THE model.config
            print("Context length:", model.config.context_length)
            max_length: int = model.config.context_length
        else:
            print(f"Context length not found in model.config, so using your default or hardcoded value. Model is {pretrained_model_name_or_path=}.")
            # max_length: int = 4  # for debugging
            max_length: int = 128  # for debugging
            # max_length: int = 256
            # max_length: int = 512
            # max_length: int = 1024
            # max_length: int = 2048 
            # max_length: int = 4096
            # max_length: int = 8192
            # max_length: int = 128_000  # ref: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
            print(f'-->{max_length=}')
    else: 
        print(f'{pretrained_model_name_or_path=}')
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32 
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True)
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right", use_auth_token=True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right", trust_remote_code=True)
        print(f'{tokenizer.pad_token=} {tokenizer.eos_token_id=}')
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token
        print(f'{tokenizer.pad_token=} {tokenizer.eos_token_id=}')
        # get context length for setting max length for training
        if hasattr(model.config, "context_length"):
            # SEEMS IT IS NOT IN THE model.config
            print("Context length:", model.config.context_length)
            max_length: int = model.config.context_length
        else:
            print(f"Context length not found in model.config, so using your default or hardcoded value. Model is {pretrained_model_name_or_path=}.")
            # max_length: int = 4  # for debugging
            max_length: int = 128  # for debugging
            # max_length: int = 256
            # max_length: int = 512
            # max_length: int = 1024
            # max_length: int = 2048 
            # max_length: int = 4096
            # max_length: int = 8192
            # max_length: int = 128_000  # ref: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
            print(f'-->{max_length=}') 
    print(f'{device=} Model device: {next(model.parameters()).device}')
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    # - Load the dataset
    print(f'-Load the dataset')
    ## Proofnet
    # dataset_val = load_dataset(path, split='validation')
    # dataset_test = load_dataset(path, split='test')
    # # Preprocess the dataset
    # if path == "hoskinson-center/proofnet":
    #     preprocess_function = preprocess_function_proofnet_simple
    #     # note: text field is usually more common!
    #     val_dataset = dataset_val.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])
    #     test_dataset = dataset_test.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, remove_columns=["nl_statement", "formal_statement"])
    ## C4
    # train_dataset = load_dataset(path='allenai/c4', name='en', split='train', streaming=True)
    # eval_dataset = load_dataset(path='allenai/c4', name='en', split='validation', streaming=True)
    # train_dataset = raw_ds_2_lm_ds_mask_eos_pad_toks(train_dataset, tokenizer, max_length)
    # eval_dataset = raw_ds_2_lm_ds_mask_eos_pad_toks(eval_dataset, tokenizer, max_length)

    # json files for putnam are not consistent and it seems they have to be: https://chatgpt.com/c/9cecca7d-d50d-42e2-b2d3-c1057bc21ef2 solve later
    # ~/putnam-math/data/Putnam_MATH_variations_static3/original/test
    # json_files = glob.glob(os.path.expanduser('~/putnam-math/data/Putnam_MATH_original_static3/test/**/*.json'), recursive=True)
    # train_dataset = load_dataset('json', data_files=json_files)
    # json_files = glob.glob(os.path.expanduser('~/putnam-math/data/Putnam_MATH_variations_static3/variations/test/**/*.json'), recursive=True)
    # eval_dataset = load_dataset('json', data_files=json_files)
    # train_dataset = raw_ds_2_lm_ds_mask_eos_pad_toks(train_dataset, tokenizer, max_length)
    # eval_dataset = raw_ds_2_lm_ds_mask_eos_pad_toks(eval_dataset, tokenizer, max_length)

    # Proofnet with 1st eos token train remaining eos not train
    from train.utils import raw_str_2_desired_af_str
    _raw_str_2_desired_af_str = lambda examples: raw_str_2_desired_af_str(examples, tokenizer)  # tokenizer needed to get eos tok to form right str to train on, max_length_not needed here.
    train_dataset = load_dataset(path, split='validation')
    eval_dataset = load_dataset(path, split='test')
    train_dataset = raw_ds_2_lm_ds_mask_eos_pad_toks(train_dataset, tokenizer, max_length, raw_str_2_desired_str=_raw_str_2_desired_af_str)
    print(f'->{len(train_dataset)=}')
    eval_dataset = raw_ds_2_lm_ds_mask_eos_pad_toks(eval_dataset, tokenizer, max_length, raw_str_2_desired_str=_raw_str_2_desired_af_str)
    print(f'->{len(eval_dataset)=}')
    # eval_dataset = None
    # print(f'->{len(train_dataset)=} {len(eval_dataset)=}')
    # max_steps: int = (len(train_dataset) * num_train_epochs) // per_device_train_batch_size  # TODO: really?

    # Training arguments
    today = date.today()
    formatted_date: str = today.strftime("%m%d%Y") # Format the date as MMDDYYYY
    output_dir_train: Path = Path(output_dir_train.format(date=formatted_date)).expanduser()
    output_dir_train.mkdir(parents=True, exist_ok=True)
    print(f'{output_dir_train=}')
    training_args = TrainingArguments(
        output_dir=output_dir_train,
        max_steps=2,  # TODO get rid of this in favour of 1 or 2 or 3 epochs
        # num_train_epochs=num_train_epochs, 
        gradient_accumulation_steps=gradient_accumulation_steps,  # based on alpaca https://github.com/tatsu-lab/stanford_alpaca, allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
        gradient_checkpointing = gradient_checkpointing,  # TODO depending on hardware set to true?
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay, 
        max_grad_norm=max_grad_norm, # TODO once real training change?
        lr_scheduler_type=lr_scheduler_type,  # TODO once real training change? using what I've seen most in vision 
        warmup_ratio=warmup_ratio,
        optim=optim,
        # logging_strategy='epoch', # TODO
        save_steps=100, # Save checkpoint every 500 steps
        save_total_limit=3, # save last 3
        logging_steps=10,  # Frequency of logging steps
        logging_first_step=True,
        logging_dir=output_dir_train,
        # evaluation_strategy='no',  # "no"`: No evaluation is done during training. no can be good to avoid memory issues.
        eval_strategy='no',  # "no"`: No evaluation is done during training. no can be good to avoid memory issues.
        # evaluation_strategy="steps",  # TODO Evaluate model at specified steps
        # eval_steps=110,  # TODO Evaluate every 100 steps
        # remove_unused_columns=False,  # TODO https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76929999#76929999 , https://claude.ai/chat/475a4638-cee3-4ce0-af64-c8b8d1dc0d90
        report_to=report_to,  # options I recommend: 'none', 'wandb'
        fp16=False,  # never ever set to True
        bf16=torch.cuda.is_bf16_supported(),
        # full_determinism=True,  # TODO periphery, Ensure reproducibility
        # torchdynamo="nvfuser",  # TODO periphery, Use NVFuser backend for optimized torch operations
        # dataloader_prefetch_factor=2,  # TODO periphery, Number of batches to prefetch
        # dataloader_pin_memory=True,  # TODO periphery, Pin memory in data loaders for faster transfer to GPU
        # dataloader_num_workers=16,  # TODO Number of subprocesses for data loading
    )

    # Initialize the Trainer 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # set to None if eval is giving you memory issues
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # Train the model
    # st()
    trainer.train()
    # st()

    # Evaluate the model
    per_device_eval_batch_size = 1
    if output_dir_test is not None:
        output_dir_test: Path = Path(output_dir_test.format(date=formatted_date)).expanduser()
        output_dir_test.mkdir(parents=True, exist_ok=True)
        eval_args = TrainingArguments(output_dir=output_dir_test, per_device_eval_batch_size=per_device_eval_batch_size, fp16=False, bf16=torch.cuda.is_bf16_supported(), report_to=report_to)
        trainer = Trainer(model=model, args=eval_args, train_dataset=None, eval_dataset=eval_dataset)
        # results: dict[str, float] = trainer.evaluate(test_dataset)
        results: dict[str, float] = eval_hf(trainer, name='', path=path, split='test')
        print(f'{path=} split=test {results=}')

    # Save the trained model
    if path_to_save_model is not None:
        model.save_pretrained(path_to_save_model)


if __name__ == "__main__":
    import time
    start_time = time.time()
    fire.Fire(main)
    print(f"Time taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
