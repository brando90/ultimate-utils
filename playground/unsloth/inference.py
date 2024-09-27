from pdb import set_trace as st

def test_unsloth_vllm(        
        max_length: int = 8192,
        use_4bit: bool = False,
    ):
    raise NotImplemented
    # TODO: https://github.com/unslothai/unsloth/issues/1063
    print('----> test_unsloth_vllm')
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = os.path.expanduser('~/data/runs/09192024_12h35m27s_run/train/checkpoint-820')
    # model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    # model_name: str = "Qwen/Qwen2-1.5B"
    print(f'{model_name=}')
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype="auto",
    #     device_map="auto"
    # )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Continue the fibonnaci sequence for a 1 step only please: 1, 1, 2, 3, 5, 8,"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    print('messages: ', messages)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print('text: ', text)
    # vllm gen
    from vllm import LLM, SamplingParams
    prompts = ["Hello, my name is"]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model=model_name)
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

def test_unsloth_inference(        
        max_length: int = 8192,
        use_4bit: bool = False,
    ):
    """ interence notebook: https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing """
    print('--> Running: test_unsloth_inference')
    import os
    from transformers import TextStreamer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    model_name = os.path.expanduser('~/data/runs/09192024_12h35m27s_run/train/checkpoint-820')
    # model_name = 'unsloth/Qwen2-1.5B'
    # model_name = 'Qwen/Qwen2-1.5B'
    print(f'{model_name=}')
    model: FastLanguageModel 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        dtype=None,  # Auto-detection for Float16/BFloat16
        load_in_4bit=use_4bit,
    )
    print(f'{type(test_unsloth_inference)=}')
    # tokenizer = get_chat_template(
    #     tokenizer,
    #     chat_template = "Qwen2-1.5B",
    #     mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    # )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    # messages = [{"role": "human", "value": "Continue the fibonnaci sequence for 1 step only please: 1, 1, 2, 3, 5, 8,"}]
    prompt = "Continue the fibonnaci sequence for 1 step only please: 1, 1, 2, 3, 5, 8,"
    print('prompt: ', prompt)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    # inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # st()
    print('text: ', text)
    # inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # inputs = tokenizer([text, text], return_tensors="pt").to(model.device)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # print(f'{inputs.size()=}')
    
    # text_streamer = TextStreamer(tokenizer)
    # st()
    # res = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 64)
    res = model.generate(**inputs, max_new_tokens=64)
    print(f'{res=}')
    completion: str = tokenizer.decode(res[0])
    # completion: str = tokenizer.decode(res)
    print(f'{completion=}')

def test_unsloth_inference_efficient(        
        max_length: int = 8192,
        use_4bit: bool = False,
    ):
    """ interence notebook: https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing """
    print('--> Running: test_unsloth_inference')
    import os
    from transformers import TextStreamer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    model_name = os.path.expanduser('~/data/runs/09192024_12h35m27s_run/train/checkpoint-820')
    print(f'{model_name=}')
    model: FastLanguageModel 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        dtype=None,  # Auto-detection for Float16/BFloat16
        load_in_4bit=use_4bit,
    )
    print(f'{type(test_unsloth_inference)=}')
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    prompt = "Continue the fibonnaci sequence for 1 step only please: 1, 1, 2, 3, 5, 8,"
    print('prompt: ', prompt)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt'
    ).to(model.device)
    print('text: ', text)
    inputs = tokenizer([text, text], return_tensors="pt").to(model.device)
    print(f'{len(inputs)=}')
    res = model.generate(**inputs, max_new_tokens=64)
    print(f'{len(res)=}')
    completion: str = tokenizer.decode(res[0])
    print(f'{completion=}')

if __name__ == "__main__":
    import fire
    import time
    print('\n-- Start')
    start_time = time.time()
    # fire.Fire(test_unsloth_vllm)
    # fire.Fire(test_unsloth_inference)
    fire.Fire(test_unsloth_inference_efficient)
    print(f"Time taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
