import sys
import glob
import os
from typing import Optional

import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM , AutoTokenizer

from multiprocessing import cpu_count

from training.reinit_models import reinit_gpt2_weights_mutates 

from pdb import set_trace as st

def seed_everything(seed: int = 0, hf_timeout: float = 5):
    """ Seed all necessary libraries to ensure reproducible results. """
    import random
    import numpy as np
    import torch
    from transformers import set_seed as hf_set_seed
    print(f'{seed=}')
    random.seed(seed)
    np.random.seed(seed)
    # Seed PyTorch (both CPU and CUDA if available)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you use multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Seed HF
    if torch.cuda.is_available():
        hf_set_seed(seed) # this gives a halting issue, so we are going to just not seed it
    else:
        print('Warning: HF is currently only dermisitic/seeded in gpu')
    # try to seed vllm
    try:
        from vllm import set_seed as vllm_set_seed
        vllm_set_seed(seed)
    except ImportError:
        print("vLLM not installed or vllm set seed has a bug, skipping vLLM seed setting.")
