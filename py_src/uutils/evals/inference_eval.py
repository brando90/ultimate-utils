import torch
from typing import Union, Callable, Optional, Any
from openai import OpenAI
import anthropic
from pathlib import Path
import sys
import os
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from collections import namedtuple

from uutils.evals.utils import batch_data
from uutils.evals.prompts_evals import SYSTEM_PROMPT_DEFAULT

from pdb import set_trace as st

# -- Generator (Inference) Classes

class Generator:
    def __init__(self):
        pass

    def __call__(self, *args, **kwards):
        pass

class HFPipelineGenerator(Generator):  
    def __init__(self, llm, sampling_params):
        super().__init__()
        from transformers import Pipeline
        self.llm: Pipeline = llm
        self.sampling_params = sampling_params

class HFDirectModelGenerator(Generator):  
    def __init__(self, llm, sampling_params):
        super().__init__()
        from transformers import Pipeline
        self.llm: Pipeline = llm
        self.sampling_params = sampling_params

class AnthropicGenerator(Generator):
    def __init__(
            self, 
            model: str , 
            sampling_params,
            api_key: str = None,
            system_prompt: str = SYSTEM_PROMPT_DEFAULT,  # You are an expert mathematician. ...
            prompt_template: Optional[str] = None, 
            verbose_init: bool = True,
            ):
        super().__init__()
        print(f'{model=}') if verbose_init else None
        if api_key is None:
            api_key = os.environ['ANTHROPIC_API_KEY'].strip()
        self.model = model
        self.sampling_params = sampling_params
        client = anthropic.Anthropic(api_key=api_key)
        self.llm = client
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.invalid_outputs = []

class UnslothGenerator(Generator):
    def __init__(
            self, 
            model: str, 
            sampling_params, 
            system_prompt: str = SYSTEM_PROMPT_DEFAULT,
            prompt_template: Optional[str] = None, 
            use_4bit: bool = False, 
            ):
        super().__init__()
        from unsloth import FastLanguageModel
        print(f'{use_4bit=}')
        mdl, tok = FastLanguageModel.from_pretrained(
            model_name=model,
            max_seq_length=sampling_params.max_length,
            dtype=None,  # Auto-detection for Float16/BFloat16
            load_in_4bit=use_4bit,
        )
        mdl: FastLanguageModel 
        self.llm = mdl
        self.tok = tok
        FastLanguageModel.for_inference(self.llm) # Enable native 2x faster inference
        self.model = model
        self.sampling_params = sampling_params
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.use_4bit = use_4bit

def before_sleep(retry_state):
    print(f"(Tenacity) Retry, error that caused it: {retry_state.outcome.exception()}")

def retry_error_callback(retry_state):
    exception = retry_state.outcome.exception()
    exception_str = str(exception)
    if "prompt is too long" in exception_str and "400" in exception_str:
        raise exception
    return 'No error that require sus to exist early.'

# After 1min 4k (due to RPM) should reset by wait 1m, much longer than e.g., 512secs = ~8mins doesn't make sense to me.
@retry(stop=stop_after_attempt(35), wait=wait_exponential(multiplier=2, max=512), 
       before_sleep=before_sleep, retry_error_callback=retry_error_callback)
def call_to_anthropic_client_api_with_retry(gen: AnthropicGenerator, prompt: str) -> dict:
    # max_tokens=8192,  # max_tokens for Claude 3.5 https://docs.anthropic.com/en/docs/about-claude/models#model-comparison
    # client = anthropic.Anthropic(api_key=gen.api_key)
    # response = client.messages.create(
    # response_text: str = gen.llm.messages.create(
    #     model=gen.sampling_params.model,
    #     max_tokens=gen.sampling_params.max_tokens,
    #     # temperature=temperature,  # note the prompt generator doesn't give this as an input
    #     system=gen.sampling_params.system,
    #     messages=[
    #         {"role": "user", "content": [{"type": "text", "text": prompt}]}
    #     ],
    #     temperature=gen.sampling_params.temperature,
    #     top_p=gen.sampling_params.top_p,
    #     n=gen.sampling_params.n,
    #     stop=gen.sampling_params.stop[:3],
    # ).content[0].text
    if not hasattr(gen.sampling_params, 'n'):
        gen.sampling_params.n = 1
    content: list[dict] = [] 
    for _ in range(gen.sampling_params.n):
        response = gen.llm.messages.create(
            model=gen.model,
            max_tokens=gen.sampling_params.max_tokens,
            system=gen.system_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            temperature=gen.sampling_params.temperature,
            top_p=gen.sampling_params.top_p,
            # n=gen.sampling_params.n,
            stop_sequences=gen.sampling_params.stop[:3],
        )
        content.append(response.content[0])
    # response = dict(content=content)
    Response = namedtuple("Response", ['content'])
    response = Response(content=content)
    # message example: https://docs.anthropic.com/en/api/messages-examples
    return response

class OpenAIGenerator(Generator):
    def __init__(
                self, 
                model: str, 
                sampling_params, 
                api_key: str = None,
                base_url: str = None,  # e.g., Mistral-7B-Instrcut-v0.2 on http://120.77.8.29:12345   
                system_prompt: str = SYSTEM_PROMPT_DEFAULT,  # You are an expert mathematician. ...
                prompt_template: Optional[str] = None, 
                verbose_init: bool = True,
                ):
        """
        export keys_brando=$(cat ~/keys/openai_api_brandos_personal_key.txt)
        # export keys_koyejolab=$(cat ~/keys/openai_api_key_brandos_koyejolab.txt)
        export OPENAI_KEY=keys_brando

        ref: https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4
            gpt-4-turbo
            gpt-3.5-turbo

        ICML math ai provides two free Baseline APIs for free calls, 
        Mistral-7B-Instrcut-v0.2 on http://120.77.8.29:12345 and 
        Llama-2-7b-chat on http://120.77.8.29:12344.  
        ref: https://docs.google.com/document/d/1tHX1IVdJ1xrN-qTLAVa0c3TmSpkSxwluW_D6rYXG8jE/edit
        ref https://www.codabench.org/competitions/2484/#/pages-tab (edited) 
        """
        super().__init__()
        # assert false if both api_key and url are Truthy -- Do not allow both options!
        assert not (api_key and base_url), f'Only one of api_key and url should be provided but got {api_key=}, {base_url=}'
        if api_key is None and base_url is not None:  # default to gpt model given
                api_key = os.environ.get("OPENAI_KEY").strip()
        # print(f'{api_key=}, {base_url=}') if verbose_init else None
        print(f'{base_url=}') if verbose_init else None
        # set attributes
        self.model = model
        self.sampling_params = sampling_params
        self.api_key = api_key
        client = OpenAI(api_key=self.api_key, base_url=base_url) 
        self.llm = client 
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.invalid_outputs = []
    
class VllmGenerator(Generator):
    def __init__(self, llm, sampling_params, lora_request=None, self.lora_adapter_path=None):
        super().__init__()
        self.llm = llm
        self.sampling_params = sampling_params
        self.lora_request = lora_request # same as bellow but defaults to None
        self.lora_adapter_path = lora_adapter_path # same as bellow but defaults to None
        self.invalid_outputs = []

class VllmLoraGenerator(Generator):
    def __init__(self, llm, sampling_params, lora_request, lora_adapter_path):
        super().__init__()
        self.llm = llm
        self.sampling_params = sampling_params
        self.lora_request = lora_request
        self.lora_adapter_path = lora_adapter_path
        self.invalid_outputs = []

class EndPointGenerator(Generator):
    def __init__(
        self,
        model: str, 
        sampling_params,
        system_prompt: str = SYSTEM_PROMPT_DEFAULT,
        ):
        super().__init__()
        self.model = model
        self.sampling_params = sampling_params
        self.system_prompt = system_prompt

def generate_with_end_point(
        gen: EndPointGenerator, 
        model: str, 
        prompt: str, 
        url: Optional[str] = 'f"http://{name}:8001/infer"',
        system_prompt: str = SYSTEM_PROMPT_DEFAULT, 
        max_tokens: int = 4096, 
        top_p: float = 0.9,
        temperature: float = 0.7,
        top_k: int = 50,
        n: int = 1,
    ) -> list[dict]:
    """ Call end point gen. """
    import requests
    responses: list[Any] = []
    assert n == 1, f'For end point for now just n=1 supported but got {n=}'
    for i in range(n):
        # print(f'{i=}')
        # prompt_sent: str = f'{system_prompt}.\n{prompt}.'
        prompt_sent: str = f'{prompt}'
        name: str = model
        url: str = url.replace('name', name)
        # print(f'{url=}')
        response = requests.post(
            url=url,
            json=dict(
                # prompt=create_gen_example(text),
                prompt=prompt_sent,
                max_new_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                top_k=top_k,
            )
        )
        # text: str = response.json()["response"]
        response_dict: dict = response.json()
        responses.append(response_dict)
    return responses

@retry(stop=stop_after_attempt(15), wait=wait_exponential(multiplier=2, max=128))
def call_to_openai_api_with_retry(gen: OpenAIGenerator, prompt: str) -> dict:
    response: dict = gen.llm.chat.completions.create(
        model=gen.model,
        messages=[
            {"role": "system", "content": gen.system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=gen.sampling_params.temperature,
        top_p=gen.sampling_params.top_p,
        n=gen.sampling_params.n,
        stop=gen.sampling_params.stop[:3],
        max_tokens=gen.sampling_params.max_tokens,
        )
    return response

@retry(stop=stop_after_attempt(15), wait=wait_exponential(multiplier=2, max=128))
def call_to_openai_client_api_with_retry(gen: OpenAIGenerator, prompt: str) -> dict:
    response: dict = gen.llm.chat.completions.create(
        model=gen.model,
        messages=[
            {"role": "system", "content": gen.system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=gen.sampling_params.temperature,
        top_p=gen.sampling_params.top_p,
        n=gen.sampling_params.n,
        stop=gen.sampling_params.stop[:3],
        max_tokens=gen.sampling_params.max_tokens,
        )
    # chat completion response format: https://platform.openai.com/docs/guides/chat-completions/response-format
    return response

def inference_vllm_prompt_only( 
        gen : Generator,
        raw_data_pt_for_prompt: list[dict], # e.g., if you want to solve nl maths this contains dict with problems {'prob':prob}, or for AF {'english': english} or {'informalization': informalization}
        prompt_template: str, # Prompt template to generate prompts from e.g., 'Solve math {$PROB}', 'Autoformalize {$ENGLISH}' etc. 
        prompt_gen_func: Callable, # needed because the prompt_template has the prompt with unique placeholders {$PLACEHOLDER_NAME} to put the names. 
        batch_size: int = 10,
        start: int = 0,  # meant for quick prototyping evals, default starts from the beginning of the eval data
        end: int = sys.maxsize,  # meant for quick prototyping evals, default grabs all eval data all the way to the end
        batched: bool = True,  # true for vllm, false (?) for hf pipeline
        ) -> dict:
        """ Do inference/gen according to prompt with data e.g., Minerva + 4 shot:
                data = {'prob': prob}
                prompt: str = prompt_gen_func(prompt_template, data['prob'])
            Note: in meta-math, ins = instruction = math problem 
            Note: return completions can be multiple strings for a single prompt e.g., useful for maj@4 voting.
        """
        # TODO: refactor to have all the objects call their own generation code internally, in an object oriented way. Abandon the attempt to do functional programming. 
        print(f'{batch_size=}')
        assert batched, f'batched should be True but got: {batched=} always batching for vllm'

        # - Form math prompts
        prompts: list[str] = [prompt_gen_func(gold_data_prob_soln, prompt_template) for gold_data_prob_soln in raw_data_pt_for_prompt]
        
        # - Get subset of eval data for quick eval prototyping
        prompts = prompts[start:end]

        # - Batch prompts
        if batched:
            assert batch_size > 0, f'batch_size should be greater than 0 but got: {batch_size=}'
            all_batched_prompts: list[list[str]] = batch_data(prompts, batch_size=batch_size)
            num_batches: int = len(all_batched_prompts)
            print(f'{num_batches=}')

        # - Return completions per prompt
        if isinstance(gen, VllmGenerator) or sinstance(gen, VllmLoraGenerator):
            from vllm import LLM, SamplingParams, RequestOutput, CompletionOutput # here otherwise warning when doing api calls in cpu laptop, vllm only works for linux 100% ref: https://github.com/vllm-project/vllm/issues/2747
            # - Generate all request outputs with completions (model solutions) for each (math) prompts
            completions: list[list[CompletionOutput]] = []
            completions_strs: list[list[str]] = []  # one completion list str per (math) prompt
            outputs: list[RequestOutput] = [] 
            for batch_idx in range(num_batches):
                batch_prompts: list[str] = all_batched_prompts[batch_idx]
                batch_outputs: list[RequestOutput] = gen.llm.generate(batch_prompts, gen.sampling_params, lora_request=self.lora_request)
                # for each output per prompt in batch of responses (let's flatten the batch)
                output: RequestOutput
                for output in batch_outputs:  
                    completions_per_prompt: list[CompletionOutput] = output.outputs
                    completions_strs_per_prompt: list[str] = [completion.text for completion in output.outputs]
                    # append completion per prompt
                    completions.append(completions_per_prompt)
                    completions_strs.append(completions_strs_per_prompt)
                    outputs.append(output)
            assert len(outputs) == len(prompts), f'Length of outputs and prompts should be equal but got: {len(outputs)=}, {len(prompts)=}'
        elif isinstance(gen, OpenAIGenerator):
            # ref: https://platform.openai.com/docs/guides/chat-completions/response-format
            # example: https://platform.openai.com/docs/guides/text-generation
            completions: list[dict] = []
            completions_strs: list[list[str]] = []
            for batch_idx in range(num_batches):
                batch_prompts: list[str] = all_batched_prompts[batch_idx]
                for prompt in tqdm(batch_prompts, total=len(batch_prompts)):
                    response: dict = call_to_openai_client_api_with_retry(gen, prompt)
                    completions.append(response)
                    comps_str_for_prompt: list[str] = [completion.message.content for completion in response.choices]  # response.choices[i].message
                    completions_strs.append(comps_str_for_prompt)
            outputs = completions
        elif isinstance(gen, AnthropicGenerator):
            # ref: https://docs.anthropic.com/en/api/messages, https://platform.openai.com/docs/guides/chat-completions/response-format
            # example: https://docs.anthropic.com/en/api/messages-examples
            serial = True
            if serial:
                completions: list[dict] = []
                completions_strs: list[list[str]] = []
                for batch_idx in range(num_batches):
                    batch_prompts: list[str] = all_batched_prompts[batch_idx]
                    for prompt in tqdm(batch_prompts, total=len(batch_prompts)):
                        response: dict = call_to_anthropic_client_api_with_retry(gen, prompt)
                        completions.append(response)
                        comps_str_for_prompt: list[str] = [content_res_obj.text for content_res_obj in response.content] # ref: https://docs.anthropic.com/en/api/messages-examples
                        completions_strs.append(comps_str_for_prompt)
                outputs = completions 
            else:
                ...
        elif isinstance(gen, HFPipelineGenerator):
            # ref: https://stackoverflow.com/a/78466524/1601580
            # note: you might get warning due to temp, top_p not being zero and sampling is false when doing beam search
            print('Note: you might get warning due to temp, top_p not being zero and sampling is false when doing beam search')
            top_p, temperature, max_length, n, num_beams = gen.sampling_params.top_p, gen.sampling_params.temperature, gen.sampling_params.max_tokens, gen.sampling_params.n, gen.sampling_params.num_beams
            do_sample: bool = True if num_beams == 1 or num_beams == None else False  # beam search doesn't need sampling take n gens from beam length(?), ref: https://stackoverflow.com/a/78466524/1601580
            truncation: bool = True if max_length is not None else False  # do truncate if max_tokens given, trucate up to max_tokens
            print(f'{num_beams=}')
            print(f'{do_sample=}, {truncation=}')
            print(f'{top_p, temperature, max_length, n, num_beams}=')
            # - Generate all request outputs with completions (model solutions) for each (math) prompts, note: batching can be bad: https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching
            completions: list[list[dict]] = []  # list when completions is a list (of dicts)
            completions_strs: list[list[str]] = []  # list when completions is a list (of strs)
            outputs: list = []  # list of outputs, 1 output per llm req
            for batch_idx in range(num_batches):
                batch_prompts: list[str] = all_batched_prompts[batch_idx]
                # for each output per prompt in batch of responses (let's flatten the batch)
                for prompt in tqdm(batch_prompts, total=len(batch_prompts)):
                    # output = pipe("This is a cool example!", do_sample=False, top_p=0.95, temperature=0.8, max_length=50, num_return_sequences=4, num_beams=5)
                    output: list[dict] = gen.llm(prompt, do_sample=do_sample, top_p=top_p, temperature=temperature, max_length=max_length, num_return_sequences=n, num_beams=num_beams, truncation=truncation)
                    completions_per_prompt: list[dict] = output
                    completions_strs_per_prompt: list[str] = [completion['generated_text'] for completion in output]
                    # append completion per prompt
                    completions.append(completions_per_prompt)
                    completions_strs.append(completions_strs_per_prompt)
                    outputs.append(completions_per_prompt)
            assert len(outputs) == len(prompts), f'Length of outputs and prompts should be equal but got: {len(outputs)=}, {len(prompts)=}'
        elif isinstance(gen, HFDirectModelGenerator):
            assert ValueError(f'Don\'t use HFDirectModelGenerator= for now, odd bug, see: https://discuss.huggingface.co/t/how-to-generate-multiple-text-completions-per-prompt-like-vllm-using-huggingface-transformers-pipeline-without-triggering-an-error/86297/4')
            import torch
            model, tokenizer = gen.llm.model, gen.llm.tokenizer
            n: int = gen.sampling_params.n
            num_beams: int = 5
            max_tokens: int = gen.sampling_params.max_tokens
            device = model.device
            # batching isn't always good in HF pipeline, ref: https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching
            # - Generate all request outputs with completions (model solutions) for each (math) prompts
            completions: list[list[dict]] = []  # list when completions is a list (of dicts)
            completions_strs: list[list[str]] = []  # list when completions is a list (of strs)
            outputs: list = []  # list of outputs, 1 output per llm req
            for batch_idx in range(num_batches):
                batch_prompts: list[str] = all_batched_prompts[batch_idx]
                # for each output per prompt in batch of responses (let's flatten the batch)
                for prompt in tqdm(batch_prompts, total=len(batch_prompts)):
                    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
                    # attention_mask = encoded_inputs['attention_mask']  # not needed since we are encoding one seq at a time, ref: https://chatgpt.com/g/g-KV0CvoH8Y-python-excellent-comments-doc-strings-types/c/cb817065-2891-4a82-bf7d-1458baa3fe36
                    completions_per_prompt: list[int] = model.generate(input_ids=input_ids, num_beams=num_beams, num_return_sequences=n, max_length=max_tokens)
                    completions_strs_per_prompt: list[str] = [tokenizer.decode(comp, skip_special_tokens=True) for comp in completions_per_prompt]
                    completions_per_prompt = outputs
                    # append completion per prompt
                    completions.append(completions_per_prompt)
                    completions_strs.append(completions_strs_per_prompt)
                    outputs.append(completions_per_prompt)
            assert len(outputs) == len(prompts), f'Length of outputs and prompts should be equal but got: {len(outputs)=}, {len(prompts)=}'
        elif isinstance(gen, EndPointGenerator):
            # from threading import ThreadPoolExecutor
            # batches_prompts: list[list[str]] = []
            # #
            # for batch_prompts in baches_prompts:
            #     with ThreadPoolExecutor(8) as executor:
            #         reqs = {}
            # - In
            batches_prompts: list[list[str]] = all_batched_prompts
            # - Out
            outputs: list[list[dict]] = [] # one request to endpoint per prompt (this case same as bellow)
            completions: list[list[dict]] = [] # one list of completions dicts (n) per prompt
            completions_strs: list[list[str]] = []  # one completion list str (n) per (math) prompt
            # for batch_prompts in tqdm(batches_prompts, total=len(batches_prompts)):
            for batch_prompts in batches_prompts:
                prompt: str
                for prompt in tqdm(batch_prompts, total=len(batch_prompts)):
                    system_prompt=''
                    name, max_tokens = gen.model, gen.sampling_params.max_tokens
                    top_p, temperature = gen.sampling_params.top_p, gen.sampling_params.temperature
                    n = gen.sampling_params.n
                    top_k = 50 # other inferences don't use this so it would have to be incorporated...TODO careful not all mdl have this
                    responses: list[dict] = generate_with_end_point(gen, name, prompt, system_prompt, max_tokens, top_p, temperature, top_k, n)
                    responses_text: list[str] = [res['response'] for res in responses]
                    #
                    outputs.append(responses)
                    completions.append(responses)
                    completions_strs.append(responses_text)
        elif isinstance(gen, UnslothGenerator):
            assert gen.sampling_params.n == 1, f'Err: Unsloth fast model only allows n=1 but it was {gen.sampling_params.n=}'
            max_tokens: int = gen.sampling_params.max_tokens
            from unsloth import FastLanguageModel
            model: FastLanguageModel = gen.llm
            tokenizer = gen.tok 
            completions: list[list[str]] = []  # list when completions is a list (of objs/strs)
            completions_strs: list[list[str]] = []  # list when completions is a list (of strs)
            outputs: list = []  # list of outputs, 1 output per llm req
            for batch_idx in range(num_batches):
                batch_prompts: list[str] = all_batched_prompts[batch_idx]
                for prompt in tqdm(batch_prompts, total=len(batch_prompts), desc='Unsloth per prompt gen loop'):
                    msg = [{"role": "system", "content": gen.system_prompt}, {"role": "user", "content": prompt}]
                    inputs = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
                    res = model.generate(inputs, max_new_tokens=max_tokens)
                    completion: str = tokenizer.decode(res[0])
                    # append completion per prompt
                    completions.append([completion])
                    completions_strs.append([completion])
                    outputs.append(completion) 
        else:
            raise ValueError(f'Unknown generator type: {gen=}')
        # - Return completions (list comp) per prompt
        assert len(completions) == len(prompts), f'Length of completions and prompts should be equal but got: {len(completions)=}, {len(prompts)=}'
        assert len(completions_strs) == len(prompts), f'Length of completions_strs and prompts should be equal but got: {len(completions_strs)=}, {len(prompts)=}'
        assert len(completions_strs) == len(completions), f'Length of completions_strs and completions should be equal but got: {len(completions_strs)=}, {len(completions)=}'
        result: dict = dict(completions=completions, completions_strs=completions_strs, outputs=outputs)
        return result

# -- Estimate for OpeanAI API inferences cost $$

def get_token_char_page_approx_equivalence():
    """
    1 tok ~ 4-5 chars e.g., hello 5, dog 3, help 4, happy 5, the 3, at 2, she 3,
    2-3 tok ~ 1 word
    4k toks = 2k words = 2000 words = 2000 / 500 = 4 pages 

    Google doc 11pt font 
    1 lines ~ 13-14 words
    1 page ~ 35-37 lines
    1 page ~ 37 lines / page * 13 words / line = 481 words / page 
    (1 char ~ 1 byte)
    """
    ...

def get_cost_inference_per_token(model: str = 'gpt-4-turbo', verbose: bool = True) -> dict:
    # gpt-4-turbo-2024-04-09 in $10.00 / 1M tokens out $30.00 / 1M tokens
    if 'gpt-4-turbo' in model:
        # to cost per token $$ / tok
        inprice: float = 10 / 1_000_000
        outprince: float = 30 / 1_000_000
        prices: dict = {'in_cost_per_tok': inprice, 'out_cost_per_tok': outprince}
        print(f'{prices=}') if verbose else None
        return prices
    else:
        raise ValueError(f'Unknown model: {model=}')

def estimate_openai_api_inference_cost(
        prompts: list[str],  # e.g., math prompts
        outputs: list[str],  # perhaps guessed to have a cost
        model: str = 'gpt-4-turbo',  # ref costs: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken#encodings
        verbose: bool = True,
        ) -> float:
    """ Estimate cost of inference for given prompts using OpenAI API. ref: https://community.openai.com/t/how-do-people-estimate-gpt4-given-that-they-changed-to-pre-paid-plan-you-dont-know-how-long-the-response-will-be/741443/3"""
    import tiktoken
    assert model in {'gpt-4-turbo', 'gpt-3.5-turbo'}, f'Unknown model: {model=}'
    assert len(prompts) == len(outputs), f'Length of prompts and outputs should be equal but got: {len(prompts)=}, {len(outputs)=}'
    # - get encoding name
    # gpt-4, gpt-3.5-turbo, text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large -> cl100k_base
    if model in {'gpt-4-turbo', 'gpt-3.5-turbo', 'text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large'}:
        encoding_name: str = 'cl100k_base'
    else: 
        raise ValueError(f'Unknown model: {model=}')
    tokenizer = tiktoken.get_encoding(encoding_name)
    cost_per_tok: dict = get_cost_inference_per_token(model)
    in_cost_per_tok, out_cost_per_tok = cost_per_tok['in_cost_per_tok'], cost_per_tok['out_cost_per_tok']
    # compute cost by going through all sentences, tokenizing multiply by cost per token, sum and then return
    print(f'number of requests/seqs to {model=}: {len(prompts)=} ')
    print(f'number of outputs of {model=}: {len(outputs)=} ')
    # for output token, use output token list (guessed) strings
    tot_in_cost, tot_out_cost = 0.0, 0.0
    for prompt, output in zip(prompts, outputs):
        # tokenize with tiktoken
        toks_in: list[int] = tokenizer.encode(prompt)
        # print(f'{toks_in=} {len(toks_in)=} {type(toks_in)=}')
        num_toks_per_in_seq: int = len(toks_in)
        toks_out: list[int] = tokenizer.encode(output)
        # print(f'{toks_out=} {len(toks_out)=} {type(toks_out)=}')
        num_toks_per_out_seq: int = len(toks_out)
        # cost per token
        in_cost_per_seq: float = num_toks_per_in_seq * in_cost_per_tok
        out_cost_per_seq: float = num_toks_per_out_seq * out_cost_per_tok
        # accumulate total cost
        tot_in_cost += in_cost_per_seq
        tot_out_cost += out_cost_per_seq
    result = {'tot_in_cost': tot_in_cost, 'tot_out_cost': tot_out_cost}
    if verbose:
        print(f'{result=}')
    return result

def estimate_tenacity_vals(model) -> dict:
    """ 
    Estimate vals for tenacity retry decorator for given model. 
    
    500 rpm = 500 requests per minute = 500 reqs / 60 sec = 8.33 requests per second
    8.33 rps
    1s (init) -> 2s (1 retry) -> 4s (2 retries) -> 8s (3 retries) -> 16s (4 retries) -> 32s (5 retries)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, max=16))

    max = max wait time in seconds.
    multiplier = number to multiply wait time after we've been rate limited.
    
    ref: https://platform.openai.com/settings/organization/limits
    ref: https://chatgpt.com/g/g-KV0CvoH8Y-python-excellent-comments-doc-strings-types/c/9c137c59-1784-4023-9e38-b1e322ede951
    """
    if model == 'gpt-4-turbo':
        rpm: int = 500
        rps: float = rpm / 60  # e.g. 8.33
    else:
        raise ValueError(f'Invalid model: {model=}')
    # estimate vals, 8.33 we can do 8.33 reqs per sec, so if we do more than that we need to wait, but we don't know the cool off
    raise NotImplementedError

# -- test

def pipeline_tests_():
    print(f'\n--> pipeline_tests_()')
    import torch
    from transformers import pipeline

    # pipe = pipeline(model="gpt2", device_map="auto", model_kwargs={"load_in_8bit": True})
    pipe = pipeline(model="gpt2", device_map="auto", model_kwargs={"load_in_4bit": True})

    # output = pipe("This is a cool example!", do_sample=True, top_p=0.95, temperature=0.8, max_length=50)
    output = pipe("This is a cool example!", do_sample=True, top_p=0.95, temperature=0.8, max_length=50, truncation=True)
    print(f'\n{output=}')
    print(f'{len(output)=}')

    output = pipe("This is a cool example!", do_sample=True, top_p=0.95, temperature=0.8, max_length=50, num_return_sequences=4, truncation=True)
    print(f'\n{output=}')
    print(f'{len(output)=}')

    output = pipe("This is a cool example!", do_sample=False, top_p=0.95, temperature=0.8, max_length=50, num_return_sequences=4, num_beams=5, truncation=True)
    print(f'\n{output=}')
    print(f'{len(output)=}')

    print()

# -- main 

def main(
        # path_2_eval_dataset: str = '~/gold-ai-olympiad/data/MATH/test',
        path_2_eval_dataset: str = '~/putnam-math/data/Putnam_MATH_original_static2/test',
        model: str = 'gpt-4-turbo',  # e.g., gpt-4-turbo, gpt-3.5-turbo
        start: int = 0, 
        end: int = sys.maxsize, 
        ):
    from data_eval_utils import get_iter_for_eval_data_set
    from prompts_evals import HELM_MATH_PROMPT_8SHOT_COT2_TEMPLATE, get_math_problem_prompt_ala_helm_8shot_cot2 
    # - Get eval data
    path_2_eval_dataset: Path = Path(path_2_eval_dataset).expanduser()
    math_gold_probs_solns: list[dict] = list(get_iter_for_eval_data_set(path_2_eval_dataset))
    math_gold_probs_solns: list[dict] = math_gold_probs_solns[start:end]
    print(f'{path_2_eval_dataset=} \n {len(math_gold_probs_solns)=}')
    assert len(math_gold_probs_solns) > 0, f'No math problems found in {path_2_eval_dataset=}'

    # - Get vllm generator
    prompt_template: str = HELM_MATH_PROMPT_8SHOT_COT2_TEMPLATE
    prompt_gen_func: Callable = get_math_problem_prompt_ala_helm_8shot_cot2
    prompts: list[str] = [prompt_gen_func(gold_data_prob_soln, prompt_template) for gold_data_prob_soln in math_gold_probs_solns]
    math_guessed_outputs: list[str] = [f"Solution: Let's think step by step. " + gold_data_prob_soln['solution'] for gold_data_prob_soln in math_gold_probs_solns]

    # - Estimate cost of inference
    result = estimate_openai_api_inference_cost(prompts=prompts, outputs=math_guessed_outputs, model=model, verbose=True)
    print(f'--> Inference cost: {result=}')

if __name__ == '__main__':
    import fire
    import time
    start = time.time()
    # main()
    # fire.Fire(main)
    fire.Fire(pipeline_tests_)
    print(f"Done!\a Time: {time.time()-start:.2f} sec, {(time.time()-start)/60:.2f} min, {(time.time()-start)/3600:.2f} hr\a")