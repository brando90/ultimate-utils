import datetime
import wandb 
import os
import torch
from pathlib import Path

from transformers import GPT2LMHeadModel, PreTrainedTokenizer, AutoTokenizer, Trainer, TrainingArguments, AutoConfig
from diversity.embeddings.div_act_based import set_random_seeds
from training.reinit_and_smaller_llama2 import get_full_llama7b
from training.utils import eval_hf_with_subsample

def eval():
    mode = 'dryrun'; seed = 0
    mode = 'online'; seed = 0
    
    # set random seed
    seed = 0
    set_random_seeds(seed)

    # - hps
    max_length = 1024

    # -- Checkpoint paths
    # https://wandb.ai/brando/beyond-scale/runs/gx2y7efl?workspace=user-brando 
    # pretrained_model_name_or_path = Path('').expanduser()
    # https://wandb.ai/brando/beyond-scale/runs/q54jr2nq/logs?workspace=user-brando
    # pretrained_model_name_or_path = Path('/lfs/ampere9/0/brando9/data/results_2024-m02-d03-t16h_32m_24s/checkpoint-1551').expanduser() 
    pretrained_model_name_or_path = Path('/lfs/ampere9/0/brando9/data/results_2024-m02-d03-t16h_32m_24s').expanduser() 

    # -- Load checkpoint
    if os.path.exists(pretrained_model_name_or_path):
        model, tokenizer = get_full_llama7b(pretrained_model_name_or_path, tokenizer_is_none=True)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')  # needed explicitly for the tokenizer sinze trainer doesn't save tokenizer
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8 else torch.float32 # if >= 8 ==> brain float 16 available or set to True if you always want fp32
        model = model.to(device)
        model = model.to(torch_dtype)
        block_size = max_length

    # -- Wandb
    output_dir = Path(f'~/data/beyond_scale/eval_results_{pretrained_model_name_or_path}/').expanduser() 
    output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.datetime.now().strftime('%Y-m%m-d%d-t%Hh_%Mm_%Ss')
    current_tmux_session = os.environ.get("TMUX", "").split(",")[-1]
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f"CUDA_VISIBLE_DEVICES = {CUDA_VISIBLE_DEVICES}")
    run_name = f'beyond scale: {today=} {pretrained_model_name_or_path=} {CUDA_VISIBLE_DEVICES=} {current_tmux_session=})'
    run = wandb.init(mode=mode, project="beyond-scale", name=run_name, save_code=True)
    print(f'{run.url=}')
    wandb.config.update({'seed': seed, 'pretrained_model_name_or_path': pretrained_model_name_or_path, 'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES, "current_tmux_session": current_tmux_session, 'output_dir': output_dir})

    # -- Eval whole datasets
    # print('---- Evaluate model on Whole OpenWebtext')
    # metrics = eval_hf_with_subsample('UDACA/pile_openwebtext2', None, 'validation', model, tokenizer, block_size, output_dir, max_eval_samples=None)
    # print(f'OpenWebtext whole: {metrics=}')

    # c4 evals first in case of errors
    print('---- Evaluate model on C4')
    max_eval_samples: int = 1024
    metrics = eval_hf_with_subsample('c4', 'en', 'validation', model, tokenizer, block_size, output_dir, max_eval_samples=max_eval_samples)
    print(f'C4 ({max_eval_samples} val samples): {metrics=}')
    print('---- Evaluate model on Whole C4')
    max_eval_samples = None
    metrics = eval_hf_with_subsample('c4', 'en', 'validation', model, tokenizer, block_size, output_dir, max_eval_samples=max_eval_samples)
    print(f'C4 whole ({max_eval_samples}): {metrics=}')

    # print('---- Evaluate model on Whole wikitext-103-v1')
    # metrics = eval_hf_with_subsample('wikitext', 'wikitext-103-v1', 'validation', model, tokenizer, block_size, output_dir, max_eval_samples=None)
    # print(f'Wikitext whole: {metrics=}')

    # -- Print config to show in log what this run was especially data set
    if mode == 'online':
        print(f'{run.url=}')
        run.finish()
    print(f'{wandb.config=}')
    print('Done!\a')

if __name__ == '__main__':
    import time
    start = time.time()
    eval()
    print(f'Done!\a Time elapsed: {(time.time() - start):.2f}secs {((time.time() - start)/60):.2f}mins {((time.time() - start)/60/60):.2f}hours\a\a')