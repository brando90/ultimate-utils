from torch import bfloat16, float16
from transformers import TrainingArguments

from uutils import expanduser
from pdb import set_trace as st


def get_training_arguments_falcon7b(output_dir="./results",
                                    per_device_train_batch_size=1,
                                    gradient_accumulation_steps=16,  # num its to accumulate before opt update step
                                    optim="paged_adamw_32bit",
                                    save_steps=10,  # how often to save, if <1 -> % of train steps
                                    logging_steps=10,  # how often to log, if <1 -> % of train steps
                                    learning_rate=2e-4,
                                    max_grad_norm=0.3,
                                    max_steps=500,  # number of training steps/its
                                    warmup_ratio=0.03,  # number of steps for a linear warmup
                                    lr_scheduler_type="constant",

                                    fp16=True,
                                    bf16=False,
                                    bnb_4bit_compute_dtype=None,  # changed it from Guanaco hf
                                    ) -> TrainingArguments:
    """ """
    if bnb_4bit_compute_dtype is None:  # only if not given try the bfloat16 default or float16 default
        from uutils.torch_uu import bfloat16_avail
        bnb_4bit_compute_dtype = bfloat16 if bfloat16_avail() else float16
        fb16 = bnb_4bit_compute_dtype == float16
        bf16 = bnb_4bit_compute_dtype == bfloat16

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
    )
    return training_arguments


def get_original_training_args(output_dir="./results",
                               per_device_train_batch_size=1,
                               gradient_accumulation_steps=16,  # num its to accumulate before opt update step
                               optim="paged_adamw_32bit",
                               save_steps=10,  # how often to save, if <1 -> % of train steps
                               logging_steps=10,  # how often to log, if <1 -> % of train steps
                               learning_rate=2e-4,
                               max_grad_norm=0.3,
                               max_steps=500,  # number of training steps/its
                               warmup_ratio=0.03,  # number of steps for a linear warmup
                               lr_scheduler_type="constant",
                               ):
    """
    """
    from transformers import TrainingArguments

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
    )
    return training_arguments


def get_training_args_falcon_7b_fp32(report_to: str = "none",
                                     output_dir="./results",
                                     per_device_train_batch_size=1,
                                     gradient_accumulation_steps=1,  # n_its to accum. before opt update step
                                     optim="paged_adamw_32bit",
                                     # optim="adafactor",
                                     save_steps=10,  # how often to save, if <1 -> % of train steps
                                     logging_steps=10,  # how often to log, if <1 -> % of train steps
                                     learning_rate=2e-4,
                                     max_grad_norm=0.3,
                                     max_steps=500,  # number of training steps/its
                                     warmup_ratio=0.03,  # number of steps for a linear warmup
                                     lr_scheduler_type="constant",

                                     verbose: bool = True,
                                     ):
    """
    original training args from Guanaco: https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing

    tricks:
        - set batch size small but grad accum large: https://chat.openai.com/share/49068fb4-1406-406e-81f5-e1f13736b0ac
            - 1 (batch size per device) * 8 (number of GPUs, from --nproc_per_node=8) * 16 (gradient accumulation steps) = 128.
        -
    """
    from uutils import get_filtered_local_params
    # decided not to do: put if stmt for dtype although says 32fb would be nice to check 16fb curious
    output_dir: str = str(expanduser(output_dir))
    get_filtered_local_params(locals(), verbose=verbose, var_name_in_front='training_arguments') if verbose else None

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        # fp16=True,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        report_to=report_to,
    )
    return training_arguments
