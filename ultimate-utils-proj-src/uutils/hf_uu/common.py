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
    for key, value in  model.state_dict().items():
        if isinstance(value, torch.Tensor):
            print(f"--> Weight '{key}' has datatype: {value.dtype}")
            if num_layers == 1:
                return
