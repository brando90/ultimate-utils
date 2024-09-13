# conda activate snap_cluster_setup
import random
from pathlib import Path
import os
import json
from typing import Union, Callable, Iterator, Optional
import datetime
import wandb
import fire

from evals.data_eval_utils import get_iter_for_eval_data_set, save_completions
from evals.prompts_evals import STOP_TOKENS, extract_answer_from_list_completion_strings_mv
from evals.prompts_evals import HELM_MATH_PROMPT_8SHOT_COT2_TEMPLATE, MATH_PROMPT_0SHOT_COT_TEMPLATE, get_math_problem_prompt_ala_helm_8shot_cot2, get_math_problem_prompt_ala_0shot_cot, HELM_MATH_PROMPT_8SHOT_COT2_TEMPLATE_MISTRAL7B_INS_V1, MATH_PROMPT_0SHOT_COT_TEMPLATE_MISTRAL7B_INS_V1
from evals.utils import extract_model_answers, eval_boxed_accuracy_results, extract_gold_answers, get_dtype_for_vllm, load_model
from evals.inference_eval import VllmGenerator, inference_vllm_prompt_only, OpenAIGenerator, HFPipelineGenerator, HFDirectModelGenerator, AnthropicGenerator, EndPointGenerator, Generator

import torch
import torch.nn

import sys
MAX_INT = sys.maxsize

from pdb import set_trace as st

def print_crucial_run_info(path_2_eval_dataset, model):
    print()
    print('---- print_crucial_run_info ----')
    print(f'----> {path_2_eval_dataset=}')
    print(f'----> {model=}')
    print(f'----> {os.environ.get("CUDA_VISIBLE_DEVICES", None)=}')
    print(f'----> {os.environ.get("CKPT_COPY", None)=}')
    print('---- print_crucial_run_info ----')
    print()

def seed_everything(seed: int, hf_timeout: float = 5):
    """
    Seed all necessary libraries to ensure reproducible results.

    Args:
        seed (int): The seed value to use.
    """
    import random
    import numpy as np
    import torch
    from transformers import set_seed as hf_set_seed
    # Seed the random module
    random.seed(seed)

    # Seed numpy
    np.random.seed(seed)

    # Seed PyTorch (both CPU and CUDA if available)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you use multi-GPU.

    # Set deterministic behavior in torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Seed Hugging Face Transformers
    if torch.cuda.is_available():
        hf_set_seed(seed) # this gives a halting issue, so we are going to just not seed it
    else:
        print('Warning: HF is currently only dermisitic/seeded in gpu')
    # # ref: https://chatgpt.com/c/a928d535-f9cb-4115-8d81-3dba18b09227
    # def set_hf_seed():
    #     import traceback
    #     try:
    #         # Attempt to set the seed using hf_set_seed, 
    #         # and if an exception occurs, print a warning, 
    #         # the exception message, and the traceback.
    #         hf_set_seed(seed)
    #     except Exception as e:
    #         print(f"Warning: Failed to set seed {seed}.")
    #         print("Exception:", e)
    #         print("Traceback:")
    #         traceback.print_exc()
    # # Run hf_set_seed in a separate thread to allow for a timeout, 
    # # and if the thread doesn't complete within the specified time, 
    # # issue a timeout warning.
    # import threading
    # thread = threading.Thread(target=set_hf_seed)
    # thread.start()
    # # # Wait for the specified timeout
    # # thread.join(hf_timeout)
    # if thread.is_alive():
    #     print(f"Warning: hf_set_seed({seed}) timed out after {hf_timeout} seconds.")
    #     # Optionally, you could terminate the thread here, though this is generally not recommended
    #     # because it might leave resources in an inconsistent state.
    #     # Signal the thread to stop

    # Seed vLLM (if applicable)
    try:
        from vllm import set_seed as vllm_set_seed
        vllm_set_seed(seed)
    except ImportError:
        print("vLLM not installed or vllm set seed has a bug, skipping vLLM seed setting.")

# -- tests

def eval_on_four_math_benchmarks_passing_gen_engine_obj(
        model,
        gen_type='vllm',
        end: int = 500,
):
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    path_2_eval_dataset: str = '~/putnam-math/data/MATH/test'
    gen = main(model=model, gen_type='vllm', path_2_eval_dataset=path_2_eval_dataset, end=end, n=1, shuffle=True, mode='dryrun')
    # doing this hack so that oom doesn't happen, re-using loaded model (even though I did try clearing up)
    path_2_eval_dataset: str = '~/putnam-math/data/OlympiadBench_Dataset/data_math_boxed_21_08_2024_v2'
    main(model=gen, gen_type='our_gen_engine', path_2_eval_dataset=path_2_eval_dataset, end=end, n=1, shuffle=True, mode='dryrun')
    path_2_eval_dataset: str = '~/putnam-math/data/Putnam_MATH_original_static_final_21_08_2024/Putnam_MATH_boxed_problems_full.json'
    main(model=gen, gen_type='our_gen_engine', path_2_eval_dataset=path_2_eval_dataset, end=end, n=1, shuffle=True, mode='dryrun')
    path_2_eval_dataset: str = '~/putnam-math/data/Putnam_MATH_variations_static_constant/test.json'
    main(model=gen, gen_type='our_gen_engine', path_2_eval_dataset=path_2_eval_dataset, end=end, n=1, shuffle=True, mode='dryrun')

def eval_on_four_math_benchmarks(
        model,  # str or mdl
        gen_type: str = 'hf_model',
        batch_size: int = 10,
        end: int = 263,
):
    path_2_eval_dataset: str = '~/putnam-math/data/MATH/test'
    main(model=model, gen_type=gen_type, path_2_eval_dataset=path_2_eval_dataset, end=end, batch_size=batch_size, n=1, shuffle=True, mode='dryrun')
    path_2_eval_dataset: str = '~/putnam-math/data/OlympiadBench_Dataset/data_math_boxed_21_08_2024_v2'
    main(model=model, gen_type=gen_type, path_2_eval_dataset=path_2_eval_dataset, end=end, batch_size=batch_size, n=1, shuffle=True, mode='dryrun')
    path_2_eval_dataset: str = '~/putnam-math/data/Putnam_MATH_original_static_final_21_08_2024/Putnam_MATH_boxed_problems_full.json'
    main(model=model, gen_type=gen_type, path_2_eval_dataset=path_2_eval_dataset, end=end, batch_size=batch_size, n=1, shuffle=True, mode='dryrun')
    path_2_eval_dataset: str = '~/putnam-math/data/Putnam_MATH_variations_static_constant/test.json'
    main(model=model, gen_type=gen_type, path_2_eval_dataset=path_2_eval_dataset, end=end, batch_size=batch_size, n=1, shuffle=True, mode='dryrun')

def main(
        # path_2_eval_dataset: str = '~/putnam-math/data/MATH/test',
        # path_2_eval_dataset: str = '~/putnam-math/data/OlympiadBench_Dataset/data_math_boxed_21_08_2024_v2',
        # path_2_eval_dataset: str = '~/putnam-math/data/Putnam_MATH_original_static_final_21_08_2024/Putnam_MATH_boxed_problems_full.json',
        path_2_eval_dataset: str = '~/putnam-math/data/Putnam_MATH_variations_static_constant/test.json',
        # -
        # path_2_eval_dataset: str = '~/putnam-math/data/Putnam_MATH_variations_static_constant/original.json',
        # -
        # model: str = 'mistralai/Mistral-7B-v0.1',
        # model: str = 'mistralai/Mistral-7B-Instruct-v0.1',
        # model: str = 'deepseek-ai/deepseek-math-7b-instruct',
        # model: str = 'meta-llama/Meta-Llama-3-8B-Instruct',
        # model: str = 'meta-llama/Meta-Llama-3-8B', 
        # model: str = 'gpt2',
        # model: str = 'gpt-3.5-turbo',
        # model: str = 'gpt-4-turbo',
        model: str = 'claude-3-5-sonnet-20240620',
        # https://docs.anthropic.com/en/api/claude-on-amazon-bedrock#api-model-names
        output_dir: Optional[str] = '~/data/results_{today}/',  # e.g., where to save completions
        completion_filename: str = 'completions.json',
        start: int = 0, 
        end: int = sys.maxsize, # Usually used to know what fraction of benchmark to evaluate on
        batch_size: int = sys.maxsize, # the size of batch size from eval set to evaluate per eval step, note: eventually evals on everything
        # batch_size: int = 5_000,  # MATH test has 5_000 
        # batch_size: int = 348,  
        n: int = 1, # num seqs to return for given prompt
        # max_tokens: int = 2048,
        # max_tokens: int = 4096,
        max_tokens: int = 8192,
        top_p: float = 0.95, 
        temperature: float = 0.8,
        # num_beams: int = 5,
        num_beams: Optional[int] = None,
        max_length: Optional[int] = None, # max input for HF/vllm models
        # gen_type: Optional[str] = None,
        # gen_type: Optional[str] = 'openai_end_point',
        # gen_type: Optional[str] = 'anthropic_end_point',
        gen_type: Optional[str] = 'vllm',
        # gen_type: Optional[str] = 'end_point',
        # gen_type: Optional[str] = 'pipeline',
        # gen_type: Optional[str] = 'hf_direct_model_gen',
        # gen_type: Optional[str] = 'our_gen_engine',
        verbose_eval: bool = True,
        # boxed_acc_probs_only: bool = False,
        boxed_acc_probs_only: bool = True,
        use_beam_search: bool = False,
        best_of: Optional[int] = None,
        mode: str = 'dryrun',  # 'dryrun' or 'online'
        # mode: str = 'online',  # 'dryrun' or 'online'
        shuffle: bool = True, 
        seed: int = 42, 
        ):
    """ """
    import time
    start_time = time.time()
    # print(f'{model=}')
    import torch
    torch.cuda.empty_cache()
    assert isinstance(path_2_eval_dataset, str), f'Err: {path_2_eval_dataset=} wrong type should be str but is {type(path_2_eval_dataset)=}.'
    print(f'\n\n-------------------------> boxed accuracy eval main() starting (with {seed=}) for {path_2_eval_dataset=}')
    seed_everything(seed)
    print_crucial_run_info(path_2_eval_dataset, model)
    # - Start wandb run
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
    current_tmux_session = os.environ.get("TMUX", "").split(",")[-1]
    today = datetime.datetime.now().strftime('%Y-m%m-d%d-t%Hh_%Mm_%Ss')
    run_name = f'MATH ({today=} {model} {path_2_eval_dataset} {CUDA_VISIBLE_DEVICES=} {current_tmux_session=})'
    run = wandb.init(mode=mode, project="Lean for MATH", name=run_name, save_code=True)
    print(f'{run.url=}')
    output_dir = Path(f'~/data/results_{today}/').expanduser() 
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'{output_dir=}')

    # - Get vllm generator
    prompt_template: str = HELM_MATH_PROMPT_8SHOT_COT2_TEMPLATE
    # prompt_template: str = MATH_PROMPT_0SHOT_COT_TEMPLATE
    # prompt_template: str = HELM_MATH_PROMPT_8SHOT_COT2_TEMPLATE_MISTRAL7B_INS_V1
    print(f'--> {prompt_template=}')
    # prompt_gen_func: Callable = get_math_problem_prompt_ala_helm_8shot_cot2
    prompt_gen_func: Callable = get_math_problem_prompt_ala_0shot_cot
    print(f'{prompt_gen_func=}')
    # extract_answer_func: Callable = extract_answer_minerva_prompt
    extract_answer_func: Callable = extract_answer_from_list_completion_strings_mv
    print(f'{extract_answer_func=}')
    # stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    stop: list[str] = STOP_TOKENS
    # push to config before loading model to avoid any common llm issues
    wandb.config.update(dict(prompt_template=prompt_template, prompt_gen_func=str(prompt_gen_func), model=model, path_2_eval_dataset=path_2_eval_dataset, output_dir=output_dir, stop_tokens=stop, extract_answer_func=extract_answer_func), allow_val_change=True)
    dtype  = get_dtype_for_vllm()
    print(f'{dtype=}')
    # sampling_params: SamplingParams = SamplingParams(n=n, max_tokens=max_tokens, top_p=top_p, temperature=temperature, stop=stop, use_beam_search=use_beam_search, best_of=best_of)
    # Note: some sampling params are not present in all inference frameworks so they need to be removed later
    from collections import namedtuple
    SamplingParams = namedtuple('SamplingParams', ['n', 'max_tokens', 'top_p', 'temperature', 'stop', 'use_beam_search', 'best_of', 'max_length', 'num_beams'])
    sampling_params = SamplingParams(n=n, max_tokens=max_tokens, top_p=top_p, temperature=temperature, stop=stop, use_beam_search=use_beam_search, best_of=best_of, max_length=max_length, num_beams=num_beams)
    print(f'{sampling_params=}')
    print(f'{model=}')
    # st()
    print(f'--> {model=} {gen_type=} {os.environ.get("CKPT_COPT", None)=}')
    if gen_type == 'openai_end_point':
        api_key = os.environ.get("OPENAI_KEY").strip()
        gen: OpenAIGenerator = OpenAIGenerator(model, sampling_params, api_key)
    elif gen_type == 'anthropic_end_point': 
        api_key = os.environ.get("ANTHROPIC_API_KEY").strip()
        gen: AnthropicGenerator = AnthropicGenerator(model, sampling_params, api_key=api_key)
    elif 'vllm' in str(gen_type).lower():
        # st()
        from vllm import LLM, SamplingParams, RequestOutput, CompletionOutput # here otherwise warning when doing api calls in cpu laptop, vllm only works for linux 100% ref: https://github.com/vllm-project/vllm/issues/2747
        llm: LLM = LLM(model=model, dtype=dtype, trust_remote_code=True)
        # remove any field not in vllm's SamplingParams code e.g., max_length is mostly a HF model concept
        default_vllm_sp_keys = vars(SamplingParams()).keys()
        _sampling_params = {key: field for key, field in sampling_params._asdict().items() if key in default_vllm_sp_keys}
        sampling_params = SamplingParams(**(_sampling_params))
        gen: VllmGenerator = VllmGenerator(llm, sampling_params)
    elif gen_type == 'pipeline':
        from transformers import pipeline, Pipeline
        mdl, tok = load_model(pretrained_model_name_or_path=model, max_length=sampling_params.max_length)
        llm: Pipeline = pipeline("text-generation", model=mdl, tokenizer=tok)
        gen: HFPipelineGenerator = HFPipelineGenerator(llm, sampling_params)
        print(f'{llm.device=}')
    elif gen_type == 'hf_direct_model_gen':
        from transformers import pipeline, Pipeline
        mdl, tok = load_model(pretrained_model_name_or_path=model, max_length=sampling_params.max_length)
        llm: Pipeline = pipeline("text-generation", model=mdl, tokenizer=tok)
        gen: HFDirectModelGenerator = HFDirectModelGenerator(llm, sampling_params)
        print(f'{llm.device=}')
        assert ValueError(f'Don\'t use {gen_type=} for now, odd bug, see: https://discuss.huggingface.co/t/how-to-generate-multiple-text-completions-per-prompt-like-vllm-using-huggingface-transformers-pipeline-without-triggering-an-error/86297/4')
    elif gen_type == 'end_point':
        gen: EndPointGenerator = EndPointGenerator(model=model, sampling_params=sampling_params)
    elif gen_type == 'hf_model':
        from transformers import pipeline, Pipeline
        mdl, tok = model
        llm: Pipeline = pipeline("text-generation", model=mdl, tokenizer=tok)
        gen: HFPipelineGenerator = HFPipelineGenerator(llm, sampling_params)
        print(f'{llm.device=}') 
    elif gen_type == 'our_gen_engine':
        assert isinstance(model, Generator)
        gen: Generator = model
    else:
        raise ValueError(f'Not support {gen_type=}')
    print(f'sampling_params:\n{sampling_params}\n{sampling_params=}')

    # - Get eval data set
    print('Get eval data set')
    path_2_eval_dataset: Path = Path(path_2_eval_dataset).expanduser()
    math_gold_probs_solns: list[dict] = list(get_iter_for_eval_data_set(path_2_eval_dataset))
    print(f'{len(math_gold_probs_solns)=}')
    # st()
    math_gold_probs_solns: list[dict] = math_gold_probs_solns[start:end]
    random.shuffle(math_gold_probs_solns) if shuffle else None
    print(f'{len(math_gold_probs_solns)=}')
    
    # filter out all dicts that don't have a latex box 
    if boxed_acc_probs_only:
        math_gold_probs_solns: list[dict] = [dict_dpt for dict_dpt in math_gold_probs_solns if isinstance(dict_dpt, dict)] 
        print(f'{len(math_gold_probs_solns)=}')
        math_gold_probs_solns: list[dict] = [dict_dpt for dict_dpt in math_gold_probs_solns if '\\boxed' in dict_dpt['solution'] or '\\fbox' in dict_dpt['solution']] 
        print(f'{len(math_gold_probs_solns)=}')
    print(f'{path_2_eval_dataset=} \n {len(math_gold_probs_solns)=}')
    assert len(math_gold_probs_solns) > 0, f'No math problems found in {path_2_eval_dataset=}'

    # - Gen completions - completions are list of lists because completions can be multiple for a single prompt, for single response completions inside are length 1
    results: dict = inference_vllm_prompt_only(gen, math_gold_probs_solns, prompt_template, prompt_gen_func, batch_size, start, end) 
    completions_strs: list[list[str]] = results['completions_strs']  # completions strs per prompt
    model_answers: list[Union[str, None]] = extract_model_answers(completions_strs, extract_answer_func)
    math_gold_answers: list[str] = extract_gold_answers(math_gold_probs_solns)
    assert len(completions_strs) == len(math_gold_probs_solns), f'Length of completions_strs and math_gold_probs_solns should be equal but got: {len(completions_strs)=}, {len(math_gold_probs_solns)=}'
    assert len(model_answers) == len(math_gold_answers), f'Length of model_answers and math_gold_answers should be equal but got: {len(model_answers)=}, {len(math_gold_answers)=}'

    # - Evaluate
    print_crucial_run_info(path_2_eval_dataset, model)
    save_completions(output_dir, completion_filename, completions_strs, model_answers, math_gold_probs_solns, math_gold_answers,)
    wandb.save(output_dir / completion_filename)
    results_d: dict = eval_boxed_accuracy_results(math_gold_answers, model_answers, verbose_eval=verbose_eval)
    print(f'{results_d["boxed_acc"]=} \n {results_d["len(results)"]=} \n {results_d["len(results_boxed)"]=} \n {results_d["sum(results_boxed)"]=}')
    wandb.log({'boxed_acc': results_d['boxed_acc'], 'len(results)': results_d['len(results)'], 'len(results_boxed)': results_d['len(results_boxed)'], 'sum(results_boxed)': results_d['sum(results_boxed)']})
  
    # - End run
    # make sampling_params a dict to save nicely in wandb or last option as a string, ref: https://chatgpt.com/c/aa91ed8e-c792-4721-8987-204a3037b4b3
    sampling_params = sampling_params._asdict() if hasattr(sampling_params, '_asdict') else sampling_params
    sampling_params = vars(sampling_params) if hasattr(sampling_params, '__dict__') else sampling_params
    try:
        sampling_params: dict = dict(sampling_params) 
    except:
        sampling_params: str = str(sampling_params) # if this fails I want to know
    wandb.config.update(dict(prompt_gen_func=str(prompt_gen_func), prompt_template=prompt_template, model=str(model), path_2_eval_dataset=path_2_eval_dataset, output_dir=output_dir, sampling_params=sampling_params), allow_val_change=True)
    print(f'{wandb.config=}')
    run.finish()
    print_crucial_run_info(path_2_eval_dataset, model)
    # # try clearing gpu mem at all costs
    # try:
    #     import gc
    #     import torch
    #     del gen; del llm
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     del mdl; del tok
    #     gc.collect()
    #     torch.cuda.empty_cache()
    # except:
    #     ...
    # return gen
    print(f"Done!\a Time: {time.time()-start_time:.2f} sec, {(time.time()-start_time)/60:.2f} min, {(time.time()-start_time)/3600:.2f} hr\a")
    return results_d

if __name__ == '__main__':
    import fire
    import time
    start_time = time.time()
    print('Running __main__')
    # main()
    fire.Fire(main)
    # pyton boxed_acc_eval.py --model meta-llama/Meta-Llama-3-8B-Instruct
    print(f"Done!\a Time: {time.time()-start_time:.2f} sec, {(time.time()-start_time)/60:.2f} min, {(time.time()-start_time)/3600:.2f} hr\a")