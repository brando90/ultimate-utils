from torch import bfloat16, float16
from transformers import TrainingArguments

from uutils import expanduser


def get_training_arguments_falcon7b(output_dir="./results",
                                    per_device_train_batch_size=4,  # todo how to set
                                    gradient_accumulation_steps=4,  # todo how to set
                                    # paging so that the sudden mem gpu spikes don't cause the run to shut down
                                    # (one non obvious cause are too long seqs)
                                    # todo: why 32 bit opt?
                                    # todo: paged nadamw opt?
                                    optim="paged_adamw_32bit",
                                    save_steps=10,
                                    logging_steps=10,
                                    learning_rate=2e-4,
                                    max_grad_norm=0.3,
                                    max_steps=500,
                                    warmup_ratio=0.03,
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
                               per_device_train_batch_size=4,
                               gradient_accumulation_steps=4,
                               optim="paged_adamw_32bit",
                               save_steps=10,
                               logging_steps=10,
                               learning_rate=2e-4,
                               max_grad_norm=0.3,
                               max_steps=500,
                               warmup_ratio=0.03,
                               lr_scheduler_type="constant",
                               ):
    """
    original training args from Guanaco: https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing
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


def get_training_args_falcon_7b_32fp_28gb_mem(report_to="none",
                                              output_dir="./results",
                                              per_device_train_batch_size=4,
                                              gradient_accumulation_steps=4,
                                              optim="paged_adamw_32bit",
                                              save_steps=10,
                                              logging_steps=10,
                                              learning_rate=2e-4,
                                              max_grad_norm=0.3,
                                              max_steps=500,
                                              warmup_ratio=0.03,
                                              lr_scheduler_type="constant",
                                              ):
    """
    original training args from Guanaco: https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing
    """
    output_dir: str = str(expanduser(output_dir))

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
