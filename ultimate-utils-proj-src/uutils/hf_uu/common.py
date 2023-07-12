import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast


def report_to2wandb_init_mode(report_to: str) -> str:
    if report_to == 'none':
        return 'disabled'
    else:
        assert report_to == 'wandb', f'Err {report_to=}.'
        return 'online'


def print_dtype_hf_model_torch(model,
                               num_layers: int = 1,
                               ):
    """Print the data type (dtype) of the weights of a Hugging Face model. e.g., is it fp16, bp16, fp32, tf32, etc."""
    print('Checking for dtype of HF model.')
    import torch
    # from transformers import AutoModel
    #
    # # Load the Hugging Face model
    # model_name = "distilbert-base-uncased"
    # model = AutoModel.from_pretrained(model_name)

    # Access the model's state dictionary
    # model_state_dict = model.state_dict()

    # Check the datatype of the weights
    # for key, value in model_state_dict.items():
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            print(f"--> Weight '{key}' has datatype: {value.dtype}")
            if num_layers == 1:
                return


def assert_model_dtype_is(dtype: torch.dtype,
                          model,
                          num_layers: int = 1,
                          ):
    """Assert that the model is of a certain data type."""
    import torch

    # Check the datatype of the weights
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert value.dtype == dtype, f'Err {value.dtype=}'
            if num_layers == 1:
                return


def debug_inference(tokenizer,
                    model: PreTrainedModel,
                    ):
    """Print a generation for debugging purposes."""
    # Inference
    prompt = f'''What is the difference between nuclear fusion and fission?
    ###Response:'''

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(
        inputs=input_ids,
        temperature=0.7,
        max_new_tokens=512,
        top_p=0.15,
        top_k=0,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


def add_pad_token_to_tokenizer_falcon(tokenizer: PreTrainedTokenizerFast,
                                      model: PreTrainedModel,
                                      pad_token: str = '[PAD]',
                                      ):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # I think this is fine if during the training pad is ignored

    # note: not implemented in generalbecause I think it's impossible to get a general word_embedding layer for any hf model :(
    # model.resize_token_embeddings(len(tokenizer))  # todo: I think this is fine if during the training pad is ignored
    # model.transformer.word_embeddings.padding_idx = len(tokenizer) - 1
    # model.config.max_new_tokens = len(tokenizer)
    # model.config.pad_token_id = len(tokenizer) - 1
    # model.config.min_length = 1
    raise NotImplementedError(
        'Not implemented in general because I think it\'s impossible to get a general word_embedding layer for any hf model :(')


def print_gpu_memory_usage():
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming that you're interested in the first GPU
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    print('----')
    print(f'Total memory: {info.total / 1024 ** 2} MB')
    print(f'Free memory: {info.free / 1024 ** 2} MB')
    print(f'Used memory: {info.used / 1024 ** 2} MB')
    print('----')
    pynvml.nvmlShutdown()


def print_gpu_name():
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming that you're interested in the first GPU
    device_name = pynvml.nvmlDeviceGetName(handle)
    print(f'GPU name: {device_name.decode("utf-8")}')
    pynvml.nvmlShutdown()


def print_visible_gpu_names():
    import pynvml
    import torch

    visible_devices = torch.cuda.device_count()

    pynvml.nvmlInit()

    for i in range(visible_devices):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        print(f'GPU {i} name: {gpu_name.decode("utf-8")}')

    pynvml.nvmlShutdown()


def get_pytorch_gpu_usage(model: PreTrainedModel = None,
                          verbose: bool = False,
                          ):
    """ Get the memory used by the model in the gpu."""
    import torch

    current_device = torch.cuda.current_device()
    # converts bytes to mega bytes
    allocated = torch.cuda.memory_allocated(current_device) / 1024 ** 2
    if verbose:
        print("-> Model device (check if in gpu):", next(model.parameters()).device)
        print(f'-> PyTorch is using: {allocated} MB on GPU {current_device}')
    return allocated


def estimate_memory_used_by_loaded_model_no_data(model: PreTrainedModel,
                                                 device: str = 'cuda',
                                                 verbose: bool = True,
                                                 ):
    """ Estimate the memory used by the model without any data. This is useful to know how many models can be loaded
    on a single gpu. Note that this is an estimate and the actual memory used by the model will be slightly higher.
    Note that this is only for the model and does not include the memory used by the tokenizer.
    """
    model.to(device)  # manually doing this to check memory used by model but hf does it in trained e.g., fsdp etc.
    allocated = get_pytorch_gpu_usage(model=model, verbose=verbose)
    return allocated
