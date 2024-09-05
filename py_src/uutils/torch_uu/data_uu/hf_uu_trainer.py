from typing import Union

import torch
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


def get_trainer_translation_tutorial(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        model,
        tokenized_books: DatasetDict,
        train_dataset=None,
        eval_dataset=None,
) -> Seq2SeqTrainer:
    """

    note: it's assumed you alread preprocessed your data & split it i.e. you already did this
    (or see our get_data_set_books_tutorial function and call it):
    books: DatasetDict = load_dataset("opus_books", "en-fr")
    print(f'{books=}')

    books: DatasetDict = books["train"].train_test_split(test_size=0.2)
    print(f'{books=}')

    print(books["train"][0])
    from uutils.torch_uu.data_uu.hf_uu_data_preprocessing import helper_get_preprocess_function_translation_tutorial
    preprocessor = helper_get_preprocess_function_translation_tutorial(tokenizer)
    tokenized_books = books.map(preprocessor, batched=True, batch_size=2)
    print(f'{tokenized_books=}')
    """
    from transformers import DataCollatorForSeq2Seq

    # Data collator that will dynamically pad the inputs received, as well as the labels.
    data_collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    """
    At this point, only three steps remain:

    - Define your training hyperparameters in Seq2SeqTrainingArguments.
    - Pass the training arguments to Seq2SeqTrainer along with the model, dataset, tokenizer, and data collator.
    - Call train() to fine-tune your model.
    """
    # from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
    # import torch
    fp16 = torch.cuda.is_available()  # True for cuda, false for cpu
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        fp16=fp16,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_books["train"],
        eval_dataset=tokenized_books["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer
