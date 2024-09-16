
"""
Original size of LLaMA v2 model: 7B parameters:
{
  "_name_or_path": "meta-llama/Llama-2-7b-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.31.0.dev0",
  "use_cache": true,
  "vocab_size": 32000
}

"""
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import torch.nn as nn
from pathlib import Path
import datasets
from datasets import load_dataset, interleave_datasets
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizer, AutoTokenizer, Trainer, TrainingArguments, AutoConfig
# from transformers import models.llama.modeling_llama.LlamaRMSNorm as LlamaRMSNorm
import math
import wandb
import os

def num_params(model: nn.Module) -> int:
    # print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    return sum(p.numel() for p in model.parameters())

def get_weight_norms(model: nn.Module, verbose: bool = False) -> None:
    """
    Prints the L1 norm of the weights of each module in the given PyTorch model.

    Args:
    model (nn.Module): The PyTorch model whose weight norms are to be printed.

    Returns:
    None
    """
    total_weight_norm: float = 0.0
    for name, module in model.named_modules():
        # Check if the module has the 'weight' attribute
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            # Calculate the L1 norm of the weights
            w_norm: float = module.weight.norm(1).item()
            total_weight_norm += w_norm
            if verbose:
                print(f"Norm of weights in module {name}: {w_norm}")
    return total_weight_norm

# -- reinit (after you've created the new arch you want)

def reinitialize_weights(model, 
                         std: float = 0.0002,  # 0.02 ref: hailey S doesn't recommend this huge value! 
                         ) -> None:
    """
    
    From cs197, we choose std = 0.02 because of these two links:
    Why we chose 0.02 for standard deviation:
    https://github.com/huggingface/transformers/blob/772307be7649e1333a933cfaa229dc0dec2fd331/src/transformers/models/llama/modeling_llama.py#L858
    https://github.com/huggingface/transformers/blob/772307be7649e1333a933cfaa229dc0dec2fd331/src/transformers/models/llama/configuration_llama.py#L127
    Default is set to 0.02 in source code (see line 858 of the first link, and 127 of hte second link)
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # nn.init.normal_(module.weight, mean=0, std=0.02)
            nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def reinitialize_weights_xavier(model):
    # """ Reinit with xavier """
    # for module in model.modules():
    #     if isinstance(module, nn.Linear):
    #         nn.init.xavier_normal_(module.weight)
    #         if module.bias is not None:
    #             nn.init.constant_(module.bias, 0)
    pass

def reinitialize_weights_kamming(model):
    """ 
    Reinit with kamming ref: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_ 

    torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
    W ~ U(-bound, bound) = 3 * sqrt(3 / fan_mode)
    fan_modoe or mode = either 'fan_in' (default) or 'fan_out'. 
    Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. 
    Choosing 'fan_out' preserves the magnitudes in the backwards pass.
    recommended to use only with 'relu' or 'leaky_relu' (default).
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif 'norm' in name.lower() or 'norm' in str(module).lower():
            nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def reinitialize_weights_gpt_neox_20B_inspired_4_llama2(model, 
                                                        L: int,  # for beyond scale we filled the data to block size which is 4096 for max seq length llama2
                                                        ):
    
    """
    Note: we nearly gpt-neox_20B (2022) & llama1 , llama2 (2019) does not say how they init

    I think gpt-neox_20B & llama2 both have pre-layernorm, because transformers without tears uses the init that gpt-neox-20B uses and llama1 says it uses prenorm,
    so maybe both pre-layernorm.
    Thus, I hope transformers without tears init/the same that transformers without tears uses works. 
    
    Init:
    FF layer: (as Wang 2021, not transformers without tears)
        -> W ~ N(0, 3/L * sqrt(D))
        decided that cuz 2021 is later than transformers without tears (2019 Nguyen, Salazer)
    Other layers (as transformers without tears(2019 Nguyen, Salazer)):
        -> W ~ N(0, sqrt(2 / (d + 4d)))
    norm_layer
        gpt-neox_20B: uses layer_norm
        llama2 uses llama1 which uses: RMSNorm (Zhang and Sennrich (2019))
        decided not to copy gpt-neox_20B (which uses W ~ N(0, sqrt(2 / (d + 4d)))) 
        because they don't share the same norm. llama1/2 use RMSnorm:
            mean_a_i = g_i * a_i / sqrt(1/n sum_j a_j^2 ) [where is eps?]
        So I decided
        -> g_i (gain) ~ constant(1)
        since we aren't training to completion so perhaps it helps at the beginning. If it diverges we can set this to small or what gpt-neox_20B uses.
        There is no offset, but I will set it to 0 in the code WLG.
    Activation:
        SwiGLU (not relu for llama1, llama2) [us for baby llama2]
        gpt-neox_20B uses...doesn't say.
    We use normal distribution because transformers without tears uses it & since gpt-neox_20B uses nearly same inits llama2 likely does too. 

    refs: rmsnorm https://arxiv.org/pdf/1910.07467.pdf
    refs: llama1 since llama2 uses same arch https://arxiv.org/pdf/2302.13971.pdf 
    ref: pytorch inits https://pytorch.org/docs/stable/nn.init.html

    ref: llama2 7b config: https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json#L13 
    ref: later https://discuss.huggingface.co/t/how-to-choose-std-for-weight-init-for-llama-2-after-reinitialize/69702

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 96, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=96, out_features=96, bias=False)
          (k_proj): Linear(in_features=96, out_features=96, bias=False)
          (v_proj): Linear(in_features=96, out_features=96, bias=False)
          (o_proj): Linear(in_features=96, out_features=96, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=96, out_features=11008, bias=False)
          (up_proj): Linear(in_features=96, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=96, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=96, out_features=32000, bias=False)
)
    return get_smaller_llama2(hidden_size=32*3, num_hidden_layers=32, verbose=verbose)
    so in_featres = 96 ==> D=96

-- NORM OF ENTIRE NET BEFORE REINITIALIZATION:
Total weight norm: 1742214.2911224365
-- NORM OF ENTIRE NET AFTER REINITIALIZATION:
Total weight norm: 19383.956434190273
Done!

some stds
7.47524945917718e-05
0.0035355339059327377
7.47524945917718e-05
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):  # all linear layers including MLP and attention, let's try this first given it's smaller
        # if 'gate_proj' == name or 'up_proj' == name or 'down_proj' == name or 'lm_head' == name:  # all FF/MLP layers (not attention)
            D = module.in_features  # I think this is right size it's xW []
            # L = module.weight.shape[1]  # I don't think you can get this from the module
            std = 3 / (L * (D)**0.5)
            nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:  # don't think biases matter cuz bias=False in all layers
                nn.init.constant_(module.bias, 0)
        # elif isinstance(module, LlamaRMSNorm):
        # if name == 'norm' or name == 'input_layernorm' or name == 'post_attention_layernorm':
        #str(model.model.layers[0].input_layernorm)
        #'LlamaRMSNorm()'
        elif str(module) == 'LlamaRMSNorm()':
            if hasattr(module, 'weight'):
                if module.weight is not None:  # todo: idk if needed for layer norm
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, 'bias'):  # I don't think RMSNorm has bias, the whole point it doesn't think centering matters so bias is similar to centering
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        else:  
            if hasattr(module, 'weight'):
                if module.weight is not None: 
                    D = module.weight.shape[0]
                    # L = module.weight.shape[1]  # I don't think you can get this from the module
                    std = (2 / (D + 4*D))**0.5  # e.g., small init attention layers
                    nn.init.normal_(module.weight, mean=0, std=std)
            if hasattr(module, 'bias'):
                if module.bias is not None:  # don't think biases matter cuz bias=False in all layers
                    nn.init.constant_(module.bias, 0)

# - get just the arch, then you have to reinit it

def get_microscopic_llama2(verbose: bool = True):
    raise NotImplementedError
    # return get_smaller_llama2(hidden_size=2, num_hidden_layers=3, verbose=verbose)

def _get_deafult_smallest_llama2_debugging(verbose: bool = True):
    return get_smaller_llama2(hidden_size=32, num_hidden_layers=1, verbose=verbose)

def get_deafult_smallest_baby_llama2_v1_36m_0p036b(verbose: bool = False, reinit: bool = True):
    """ 
    with hps: 
        hidden_size=32, num_hidden_layers=32
    num_params = 35_997_728

    Starting to reinitialize the model...
    Original number of parameters: 35997728
    -- NORM OF ENTIRE NET BEFORE REINITIALIZATION:
    Total weight norm (before): 576430.1846704483
    -- NORM OF ENTIRE NET AFTER REINITIALIZATION:
    Total weight norm (after): total_weight_norm_after_reinit=7483.21137085557
    """
    print('Warning: you might need to reinit the weights if your using baby llama2.')
    return get_smaller_llama2(hidden_size=32, num_hidden_layers=32, verbose=verbose, reinit=reinit)

def get_deafult_smallest_baby_llama2_v2_109m_0p109(verbose: bool = False, reinit: bool = True):
    """  
    with hps:
        hidden_size=32*3, num_hidden_layers=32
    num_params = 108_779_616
    109M ~ 108_779_616 
    """
    return get_smaller_llama2(hidden_size=32*3, num_hidden_layers=32, verbose=verbose, reinit=reinit)

def get_max_context_length(model):
    # get context length for setting max length for training
    if hasattr(model.config, "context_length"):
        print("Context length:", model.config.context_length)
        max_length = model.config.context_length
    else:
        max_length = 4096
    block_size: int = 4096
    return block_size

def get_full_llama7b(pretrained_model_name_or_path: str = "meta-llama/Llama-2-7b-hf", put_model_on_device: bool = False, tokenizer_is_none: bool = False):
    torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8 else torch.float32 # if >= 8 ==> brain float 16 available or set to True if you always want fp32
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, torch_dtype=torch_dtype, use_auth_token=True)
    if tokenizer_is_none:
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right", use_fast=False, trust_remote_code=True, use_auth_token=True)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token
        print(f'{tokenizer.pad_token=} {tokenizer.eos_token_id=}')
    if put_model_on_device:
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = model.to(torch_dtype)
    return model, tokenizer

def get_full_llama7b_reinit(L: int, gpu_idx: int = -1, reinit_type: str = 'reinitialize_weights_gpt_neox_20B_inspired_4_llama2'):
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="auto")
    model = AutoModelForCausalLM.from_config(config)
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True, torch_dtype=torch.bfloat16, use_auth_token=True,)
    if gpu_idx >= 0:
        device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8 else torch.float32 # if >= 8 ==> brain float 16 available or set to True if you always want fp32
        model = model.to(torch_dtype)
    if reinit_type == 'reinitialize_weights_gpt_neox_20B_inspired_4_llama2':
        reinitialize_weights_gpt_neox_20B_inspired_4_llama2(model, L=L)
    return model

def get_smaller_llama2(hidden_size : int = 2048, 
                       num_hidden_layers : int = 12, 
                       return_tokenizer: bool = False, 
                       verbose : bool = False,
                       reinit: bool = True,
                       L: int = 4096,
                       ):
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    config.hidden_size = hidden_size
    config.num_hidden_layers = num_hidden_layers
    smaller_model = AutoModelForCausalLM.from_config(config)
    if reinit:
        reinitialize_weights_gpt_neox_20B_inspired_4_llama2(smaller_model, L=L)
    # NOTE: putting torch_dtype in the config doesn't work, so you have to move the model to bfloat16 later with model.to(torch.bfloat16)
    # torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8 else torch.float32 # if >= 8 ==> brain float 16 available or set to True if you always want fp32
    # device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    # smaller_model = smaller_model.to(device)
    # smaller_model = smaller_model.to(torch_dtype)
    # print(f'Model is currently on: {next(iter(smaller_model.parameters())).dtype=}')
    if verbose:
        print(f'config: {config}')
        print("Smaller number of parameters:", sum(p.numel() for p in smaller_model.parameters()))
        print(f'Model is currently on: {next(iter(smaller_model.parameters())).device=}')
        print(f'Model is currently on: {next(iter(smaller_model.parameters())).dtype=}')
        print()
    if return_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', padding_side="right", use_fast=False, trust_remote_code=True, use_auth_token=True)
        return smaller_model, tokenizer
    return smaller_model

# ---- Baby Llama v2 lets fix it with efficient net

def get_baby_llamav2_36m(verbose: bool = False, reinit: bool = True):
    # return get_smaller_llama2(hidden_size=32, num_hidden_layers=32, verbose=verbose, reinit=reinit)
    raise NotImplemented

# ---- Tests

# def _test_generate_smaller_model():
#     """
#     ref: https://stackoverflow.com/questions/76971761/how-to-adapt-llama-v2-model-to-less-than-7b-parameters
#     """
#     print('Starting to generate a smaller model...')
#     # Load the pretrained LLaMA v2 config
#     config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
#     print(f'config: {config} {type(config)}')
#     print()
#     # Print the original number of parameters 
#     model = AutoModelForCausalLM.from_config(config) 
#     print("Original number of parameters:", sum(p.numel() for p in model.parameters()))

#     # Modify the config to reduce size
#     config.hidden_size = 2048
#     config.num_hidden_layers = 12

#     # Create a new smaller model from the modified config
#     smaller_model = AutoModelForCausalLM.from_config(config)
#     print("New number of parameters:", sum(p.numel() for p in smaller_model.parameters()))

def _test_reinit_model():
    """ 
export CUDA_VISIBLE_DEVICES=6
    """
    torch.cuda.empty_cache() 
    print('Starting to reinitialize the model...')
    
    # - Get smaller llama2 model
    # model = get_deafult_smallest_llama2()
    # model = get_deafult_smallest_baby_llama2_v1_36m_0p036b()
    # model = get_deafult_smallest_baby_llama2_v2()
    # model = get_full_llama7b_reinit(L=4096)
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # - sanity checks
    print(f'Model is currently on: {next(iter(model.parameters())).device=}')
    print(f'Model is currently on: {next(iter(model.parameters())).dtype=}')
    print("Original number of parameters:", sum(p.numel() for p in model.parameters()))
    # - Check norm before reinitialization
    print("-- NORM OF ENTIRE NET BEFORE REINITIALIZATION:")
    total_weight_norm_before_reinit = get_weight_norms(model)
    print(f"Total weight norm (before): {total_weight_norm_before_reinit}")
    # - Reinitialize weights
    # reinitialize_weights(model)
    # reinitialize_weights_kamming(model)
    reinitialize_weights_gpt_neox_20B_inspired_4_llama2(model, L=4096)
    print("-- NORM OF ENTIRE NET AFTER REINITIALIZATION:")
    total_weight_norm_after_reinit = get_weight_norms(model)
    print(f"Total weight norm (after): {total_weight_norm_after_reinit=}")
    assert total_weight_norm_before_reinit != total_weight_norm_after_reinit, f'Error: total_weight_norm_reinit == total_weight_norm' 
    assert total_weight_norm_before_reinit > total_weight_norm_after_reinit, f'Error norm before reinit < norm after reinit (should be smaller after reinit).'

if __name__ == '__main__':
    import time
    start = time.time()
    print()
    _test_reinit_model()
    print('Done!\a\a\a')
