"""
todo:
    - finish passing the HF block_size tokenization code here so its modular
    - add function to our train code train.py
    - print the sequence length of the data once we include this code
    - create a unit test here to test block size
    - use the re-init code smart ally & brando wrote
"""
from itertools import chain
import math
import random
from typing import Optional, Any, Callable

import torch

import datasets
from datasets import load_dataset, interleave_datasets, IterableDataset, Dataset

from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoConfig
from transformers.testing_utils import CaptureLogger
from transformers import GPT2Tokenizer

import multiprocessing

def cuda_debug():
    import torch

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    # Get the CUDA version used by PyTorch
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")

    # Get the number of CUDA devices (GPUs)
    num_cuda_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_cuda_devices}")

def do_quick_matrix_multiply():
    """
python -c "import torch; print(torch.randn(2, 4).to('cuda') @ torch.randn(4, 1).to('cuda'));"
    """
    print(torch.randn(2, 4).to('cuda') @ torch.randn(4, 1).to('cuda'))

def get_actual_data_batch(data_set_or_batch):
    """ Returns the actual  data batch from the HF dataset obj e.g., dataset, batch etc. """
    data_batch = next(iter(data_set_or_batch))
    return data_batch

def get_vocab_size_and_ln(tokenizer: GPT2Tokenizer) -> tuple[int, float]:
    """
    Calculate the vocabulary size and its natural logarithm for a given tokenizer.

    Note:
        Sanity check -- is loss random? lnV = -ln(1/V) = -ln(1/50257) = 10.82 since CE = avg_i v_i * ln(1/p_i) but only one token is right so vi = 1 for some i so CE = ln(1/p_i)

    Args:
    tokenizer (GPT2Tokenizer): A tokenizer from the Hugging Face library.

    Returns:
    tuple[int, float]: A tuple containing the vocabulary size and its natural logarithm.
    """
    vocab_size = len(tokenizer)  # Get the size of the tokenizer's vocabulary
    ln_vocab_size = math.log(vocab_size)  # Calculate the natural logarithm of the vocabulary size
    return vocab_size, ln_vocab_size

def num_tokens(max_steps: int, batch_size: int, L: int, num_batches: int) -> int:
    """
    All sequences are of length L, due to our block size code. 
    num_batch = when using distributed training. 
            num_tokens_trained = max_steps * batch_size * L * num_batches

    how long do I have to train     
    """
    num_tokens_trained = max_steps * batch_size * L * num_batches
    return num_tokens_trained

def get_freest_gpu():
    # Get the index of the GPU with the most free memory
    devices = list(range(torch.cuda.device_count()))
    free_memory = [torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device) for device in devices]
    freest_device = devices[free_memory.index(max(free_memory))]
    return freest_device

# Use for IterableDatasetDict objects (i.e. streaming=T, split is unspecified (each key in dict is name of split))
def view_exs_iterable_dataset_dict(dataset, num_exs=10, split='train'):
  dataset_split = dataset[split]
  for ex in dataset_split:
    print(ex)
    print('example details: keys', ex.keys(), ', text char length', len(ex['text']), '\n---')
    num_exs -= 1
    if num_exs == 0:
      break

# Use for IterableDataset objects (i.e. streaming=T, split=specified)
def view_exs_iterable_dataset(dataset_split, num_exs=10):
  for ex in dataset_split:
    print(ex)
    print('example details: keys', ex.keys(), ', text char length', len(ex['text']), '\n---')
    num_exs -= 1
    if num_exs == 0:
      break

def get_lm_examples_1st_eos_mask_remaining_eos(        
        examples,
        tokenizer: AutoTokenizer, 

        remove_to_long_seqs: bool = False,
        debug: bool = False,
        ) -> dict[str, torch.Tensor]:
    """ 
    Train only on first occurence of eos. The remaining eos are masked out. If 
        - train up to 1st ocurrence of eos token, mask out the rest of the eos tokens.
        - drop or not seqs that are too long, i.e., have no eos token.
    Assumes: pad == eos
    
    ref: https://discuss.huggingface.co/t/why-does-the-falcon-qlora-tutorial-code-use-eos-token-as-pad-token/45954/13?u=brando 
    ref: https://chat.openai.com/share/02d16770-a1f3-4bf4-8fc2-464286daa8a1
    ref: https://stackoverflow.com/questions/76633368/how-does-one-set-the-pad-token-correctly-not-to-eos-during-fine-tuning-to-avoi
    """
    # - Get lm example
    seq_length: int = examples['input_ids'].size(0)
    print(f'{examples["input_ids"].size()=}, {seq_length=}') if debug else None
    examples["labels"] = examples["input_ids"].clone()  # labels is hardcoded in HF so put it!
    eos_token_id = tokenizer.eos_token_id
    print(f'{tokenizer.pad_token_id=}, {tokenizer.eos_token_id=}') if debug else None
    seqs_to_drop: list[int] = [] # store idx to drop (to long), we don't want to modify the two lists at the same time as we are looping through them
    for idx, input_ids in enumerate(examples["input_ids"]):
        assert len(input_ids.size()) == 1, 'Should be 1 example at a time in the batch or set of examples (e.g., 1 of 1000 if batched=True).'
        # Find all occurrences of eos_token
        _eos_positions = (input_ids == eos_token_id).nonzero() # get idxs for eos as a [z, 1] tensor or [z, n] for n dim tensors since all idxs for eos are needed to locate it. 
        eos_positions = _eos_positions[:, 0] # get the [z] values for eos
        if eos_positions.nelement() > 0:  # Check if eos_token is present --> if yes then make sure to trian on only it, att mask eos after and not train on those eos's either
            first_eos_position = eos_positions[0]
            examples["attention_mask"][idx, first_eos_position] = 1  # Set the mask value to 1
            examples["attention_mask"][idx, first_eos_position + 1:] = 0 
            assert examples["labels"][idx, first_eos_position] == eos_token_id, f"The label for the first eos_token is not eos, it's {eos_token_id=}"
            # after first eos token mask everything (eos AND pad, hopefully that's all there but we can sanity check later)
            examples["labels"][idx, first_eos_position + 1:] = -100
        elif remove_to_long_seqs:
            assert eos_positions.nelement() == 0, 'Error: there should be no eos if this if stmt is exexuted.'
            # record to drop this seq, has no eos so too long + flag says to drop it
            seqs_to_drop.append(idx)
        else:
            # no eos should be present at all
            assert sum(examples["labels"][idx] == -100) == 0, f"If it's too long then no pad as eos token present but got some pad token to ignore incorrectly:\n{examples['labels']=}"
            assert sum(examples["attention_mask"][idx] == 0) == 0, f"If it's too long then no pad as eos token present but got some pad token to ignore incorrectly:\n{examples['input_ids']=}"
            pass # nop: no eos in seq so too long, but keep it for training anyway
    # -- Drop seqs with no eos
    if seqs_to_drop:
        examples["input_ids"] = torch.stack([input_ids for idx, input_ids in enumerate(examples["input_ids"]) if idx not in seqs_to_drop])
        examples["attention_mask"] = torch.stack([mask for idx, mask in enumerate(examples["attention_mask"]) if idx not in seqs_to_drop])
        examples["labels"] = torch.stack([labels for idx, labels in enumerate(examples["labels"]) if idx not in seqs_to_drop])
    return examples

def raw_dataset_2_lm_data( 
        dataset: Dataset,
        tokenizer: Optional[AutoTokenizer], 
        max_length: Optional[int],
        # e.g: {'text': examples['text']} or preprocess str to get what you need {'text': f"[ex['nl'] ex['fl'] {tok.eos_token}]" for ex in examples} e.g., combine right cols + add eos.
        raw_str_2_desired_str: Optional[callable] = None, # good default lambda examples: {'text': examples[desired_dataset_column]} if unsure + make sure eos for all examples 
        desired_dataset_column: str = 'text', 

        method_to_remove_columns: str = 'keys',

        padding: str = 'max_length',
        truncation: bool = True, 
        return_tensors: str = 'pt',

        batched: bool = True, # Setting `batched=True` in the `dataset.map` function of Hugging Face's datasets library processes the data in batches rather than one item at a time, significantly speeding up the tokenization and preprocessing steps.
        streaming: bool = False,

        format: str = 'torch',
        get_lm_examples_function: Callable = get_lm_examples_1st_eos_mask_remaining_eos,
        num_proc: int = (6 * multiprocessing.cpu_count()) // 8, # None for no .map mp
        # num_proc: int = None # None for no .map mp
        ):
    print(f'{num_proc=}, {multiprocessing.cpu_count()=}')
    # - Step 1: Get desired string data set (.map for strings)
    if raw_str_2_desired_str is not None:
        desired_examples_str_dataset = dataset.map(raw_str_2_desired_str, batched=batched, num_proc=num_proc) # note: we can't remove all str columns here or we will remove the ones we want to tokenize by accident

    # - Step 2: Get tokenized data set (.map for tok)
    desired_examples_str_dataset = desired_examples_str_dataset.with_format(format)  # annoying that return tensors in the tokenizer on it's own doesn't put it into a pt tensor, so for now we keep both.
    remove_str_columns = get_column_names(desired_examples_str_dataset, streaming, method_to_remove_columns)  # remove all keys that are not tensors to avoid bugs in collate function in task2vec's pytorch data loader
    tokenize_function = lambda examples: tokenizer(examples[desired_dataset_column], padding=padding, max_length=max_length, truncation=truncation, return_tensors=return_tensors)
    tokenized_datasets = desired_examples_str_dataset.map(tokenize_function, batched=batched, remove_columns=remove_str_columns, num_proc=num_proc)

    # - Step 3: Get lm data set (finally right eos, padding etc.)
    # get_lm_examples_function = lambda examples : group_texts(examples, block_size)
    # get_lm_examples_function = lambda examples: get_lm_examples_1st_eos_mask_remaining_eos(examples, tokenizer)
    lm_dataset = tokenized_datasets.map(get_lm_examples_function, batched=batched, num_proc=num_proc)
    return lm_dataset

def get_size_of_seq_len(dataset_or_batch, verbose: bool = True, streaming: bool = True, batch_size: int = 2) -> int:
    """Print size of a sequence length in a batch. Give a hf data set obj (batches are data set objs sometimes)."""
    batch = get_data_from_hf_dataset(dataset_or_batch, streaming=streaming, batch_size=batch_size)
    size_seq_len = len(next(iter(batch))["input_ids"])
    if verbose:
        print(f'{size_seq_len=}')
        print(f'{len(next(iter(batch))["input_ids"])=}')
    return size_seq_len

def get_column_names(dataset, 
                    #   split: str = 'train',
                      method: str = 'keys', 
                      streaming: bool = True,
                      ):
    if method == 'features':
        # column_names = list(dataset[spit].features)
        column_names = list(dataset.features)
    elif method == 'keys':
        batch = get_data_from_hf_dataset(dataset, streaming=streaming, batch_size=1)
        column_names = next(iter(batch)).keys()
        # column_names = next(iter(dataset)).keys()
    else:
        raise ValueError(f"method {method} not supported")
    return column_names

def get_data_from_hf_dataset(dataset, 
                             streaming: bool = True, 
                             batch_size: int = 4, 
                            #  shuffle: bool= False, # shuffle is better but slower afaik
                            #  seed: int = 0, 
                            #  buffer_size: int = 500_000,
                             ):
    """ Gets data from a HF dataset, it's usually an iterator object e.g., some ds.map(fn, batched=True, remove_columns=remove_columns) has been applied. 
    Handles both streaming and non-streaming datasets, take for streaming and select for non-streaming.
    """
    # streaming = ataset.streaming  # TODO perhaps improvement
    # sample_data = dataset.select(range(batch_size)) if not isinstance(dataset, datasets.iterable_dataset.IterableDataset) else dataset.take(batch_size)
    batch = dataset.take(batch_size) if streaming else dataset.select(random.sample(list(range(len(dataset))), batch_size))
    return batch


def tokenize_function(examples, tokenizer, text_column_name: str):
    """ 
    creates a tokenize function that can be used in HF's map function and you specify which text column to tokenize.
    
    Assumes batched=True so examples is many row/data points.
    """
    return tokenizer(examples["text_column_name"])

def preprocess(examples, tokenizer, max_length: int = 1024):
    return tokenizer(examples["text"], padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    # return tokenizer(examples["text"], padding="max_length", max_length=model.config.context_length, truncation=True, return_tensors="pt")

def group_texts(examples, # if batched=True it's a dict of input_ids, attention_mask, labels of len(examples['input_ids']) = 1000 
                block_size: int,  # 4096, 1024
                ):
    """
    tokenizer = ...obtained from your model... 
    tokenize_function = lambda examples: tokenize_function(examples, tokenizer=tokenizer) 
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=column_names)

    if used as above then examples is
    examples = {'input_ids': [[...], [...], ...], 'attention_mask': [[...], [...], ...], 'labels': [[...], [...], ...]]]}
    examples.keys() = dict_keys(['input_ids', 'attention_mask'])
    type(examples) = <class 'dict'>
    type(examples['input_ids']) = <class 'list'>
    len(examples['input_ids']) = 1000  # if batched=True

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map    
    """
    # Concatenate all texts, usually 1000 since batched=True leads to len(examples['input_ids']) = 1000
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size  # rounds down
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def group_texts_v2(examples, # if batched=True it's a dict of input_ids, attention_mask, labels of len(examples['input_ids']) = 1000 
                block_size: int,  # 4096, 1024
                ):
    """
    tokenizer = ...obtained from your model... 
    tokenize_function = lambda examples: tokenize_function(examples, tokenizer=tokenizer) 
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=column_names)
    _group_texts = lambda examples : group_texts_v2(examples, block_size)
    lm_train_dataset = tokenized_train_datasets.map(_group_texts, batched=True)

    if used as above then examples is
    examples = {'input_ids': [[...], [...], ...], 'attention_mask': [[...], [...], ...], 'labels': [[...], [...], ...]]]}
    examples.keys() = dict_keys(['input_ids', 'attention_mask'])
    type(examples) = <class 'dict'>
    type(examples['input_ids']) = <class 'list'>
    len(examples['input_ids']) = 1000  # if batched=True

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder for each of those groups of 1,000 texts. 
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map    
    """
    # Concatenate all texts for each key in the examples e.g., it creates one concatenated list of all input_ids, one for all attention_mask, etc.
    # for column_name in examples.keys():
    #     # chain makes an iterator that returns elements from each iterator in order, basically concatenates iterators 
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # # get sequences of length block_size, then add eos token to end of each sequence and mask the rest of the sequence
    # result = {}
    # for k, t in concatenated_examples.items():
    #     # Initialize a list for each key (really key="text" is the one we care about) in the result
    #     result[k] = []
    #     total_length = len(t)  # Assuming t is a list or has a length
    #     for i in range(0, total_length, block_size):
    #         # Append the sublist of t from i to i + block_size
    #         seq = t[i : i + block_size]

    #         result[k].append(t[i : i + block_size])
    result["labels"] = result["input_ids"].copy()
    return result

# -- eval code

def compute_metrics(eval_preds):
    """ todo document clearly, from SS's code. """
    import evaluate
    metric = evaluate.load("accuracy")
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)

def eval_hf(trainer: Trainer, path: str, name: str, split: str, max_eval_samples: Any = 'Unknown_Eval_Max_Samples',):
    metrics = trainer.evaluate()
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity
    path = path.replace('/', '_')  # needed only when saving results
    print(f'Eval metrics {path} {name} {split} {max_eval_samples}: {metrics=}')
    trainer.log_metrics(f"eval_{path}_{name}_{split}_{max_eval_samples}", metrics)  # display metrics
    trainer.save_metrics(f"eval_{path}_{name}_{split}_{max_eval_samples}", metrics)
    return metrics

def eval_hf_with_subsample(path, name, split, model, tokenizer, block_size, output_dir, 
                           max_eval_samples: int = 1024,
                           streaming: bool = True, 
                           verbose: bool = True,
                           print_str: Optional[str] = None,
                           per_device_eval_batch_size: int = 1,  # recommended size is the train batch size you are using, since if the trainer.trian() worked with no OMM issues, likely trainer.evaluate() will work too with same batch size
                           ):
    eval_dataset = load_dataset(path, name, streaming=streaming, split=split).with_format("torch") 
    eval_dataset2 = raw_dataset_2_lm_data(eval_dataset, tokenizer, block_size)
    if max_eval_samples is None:
        eval_batch2 = eval_dataset2 
    else:
        eval_batch2 = eval_dataset2.take(max_eval_samples)
    print(f'Saving eval results at: {output_dir=}') # The output directory where the model predictions and checkpoints will be written.
    eval_args = TrainingArguments(output_dir=output_dir, per_device_eval_batch_size=per_device_eval_batch_size, fp16=False, bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8)
    trainer = Trainer(model=model, args=eval_args, train_dataset=None, eval_dataset=eval_batch2)
    metrics = eval_hf(trainer, path, name, split,)
    if verbose:
        print(f'----> {path=}, {name=}, {split=}, {metrics=}, {max_eval_samples=}')
    if print_str is not None:
        print(print_str)
    return metrics

if __name__ == "__main__":
    from time import time
    start_time = time()
    print(f"Done!\a Total time: {time() - start_time} seconds, or {(time() - start_time)/60} minutes. or {(time() - start_time)/60/60} hours.\a")