"""
Goal: making HF training script for model (e.g., llama v2). 

Inspiration:
- ref: The unreasonable effectiveness of few-shot learning for machine translation https://arxiv.org/abs/2302.01398
- ref: colab: https://colab.research.google.com/drive/1io951Ex17-6OUaogCo7OiR-eXga_oUOH?usp=sharing
- ref: SO on collate: https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76929999#76929999
"""
from pathlib import Path
import datasets
from datasets import load_dataset, interleave_datasets
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizer, AutoTokenizer, Trainer, TrainingArguments, AutoConfig
import math


# -- Tests

# -- Experiments 

def train():
    # feel free to move the import statements if you want, sometimes I like everything in one place so I can easily copy-paste it into a script
    from pathlib import Path
    import datasets
    from datasets import load_dataset, interleave_datasets
    import torch
    import transformers
    from transformers import PreTrainedTokenizer
    from transformers import GPT2LMHeadModel, PreTrainedTokenizer, AutoTokenizer, Trainer, TrainingArguments, AutoConfig

    import math
    
    # transformers.logging.set_verbosity_info()  # uncomment for debugging

    # -- Load model and tokenizer  # TODO: change to llama v2
    config = AutoConfig.from_pretrained("gpt2")
    context_length = config.max_position_embeddings  # perhaps you can query first layer from attention input matrix
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # name = "tiiuae/falcon-rw-1b",

    # - Ensure padding token is set TODO: how does this not screw up the fine-tuning? e.g., now model doesn't learn to predict eos since it's padded our by mask, ref: https://discuss.huggingface.co/t/why-does-the-falcon-qlora-tutorial-code-use-eos-token-as-pad-token/45954
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Padding token is not set.")

    # -- Load datasets
    # - Get train data set
    # path, name = 'brando/debug0_af', 'debug0_af'
    path, name  = ['c4', 'wikitext'], ['en', 'wikitext-103-v1']
    # train_dataset = load_dataset(path, name, streaming=True, split="train").with_format(type="torch")
    train_datasets = [load_dataset(path, name, streaming=True, split="train").with_format("torch") for path, name in zip(path, name)]
    probabilities = [1.0/len(train_datasets) for _ in train_datasets]  # TODO: perhaps we should change weights to informal and formal have same weight? right now is just in terms of list of data sets perhaps having 2 interleaves one for formal one for informal then use another interleave and do 50/50?. 
    train_dataset = interleave_datasets(train_datasets, probabilities)
    # TODO: suffle data set False, True, note i've experienced that with shuffle_ds.take(512) is slow...

    # - Get eval data set (AF for us)
    per_device_eval_batch_size = 8  # TODO: change to something larger, right now due to size of my debug0
    # TODO: probably need to write a collate_fn for the eval so that the eval is done right?
    # TODO: we need ppl (and ideally token edit distance for eval, reason explained here: https://arxiv.org/abs/2304.15004)
    # eval_dataset = load_dataset(path, name, streaming=False, split="test").with_format(type="torch") 
    eval_dataset = train_dataset  # TODO: fix obviously to something else using af

    # -- Compute max steps
    per_device_train_batch_size = 512
    dataset_size = int(1.5e12)  # TODO, doesn't seem easy to solve. Either count all the sequennces/rows or have the meta data have this. Or make this number huge. 
    # TODO dataset.info['split']['train']['num_examples'
    # dataset_size = sum(len(dataset) for dataset in datasets)  # TODO: works on with streaming = False?
    # dataset_size = sum(dataset.cardinality() for dataset in datasets)
    print(f'{dataset_size=}')
    # # TODO: feel free to fix the issue if I'm not seeing all the data points...
    # per_device_train_batch_size = batch_size
    num_epochs = 1
    max_steps = (dataset_size // per_device_train_batch_size) * num_epochs
    print(f'{max_steps=}')
    ## DOESNT WORK num_train_epochs = 3  # TODO: since I decided to do streaming = False and if we collect enough data it's unlikely we see it all hopefully (if we do 3 times seems good given that LLMs are trained to see the data only once this seems a sensible soln, + in the imagenet days things were trained to convergence with no overfitting ref: https://arxiv.org/abs/1801.00173)

    # -- Define custom collate function
    def custom_collate_fn(data: list[dict[str, str]], tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
        # ref: https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76929999#76929999
        # - Ensure tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # - Extract and concatenate informal and formal statements
        # Demos how to handle data form HF that has different columns
        sequences: list[str] = []
        for idx, example in enumerate(data):
            # # Handle null values
            # informal = example.get("generated informal statement", "") or ""
            # formal = example.get("formal statement", "") or ""

            # # Skip if both are empty
            # if not informal and not formal:
            #     continue

            # sequences.append(f'informal statement {informal} formal statement {formal}')

            # Retrieve the value for "text" from the dictionary or default to an empty string if not present or falsy. ref: https://chat.openai.com/share/bead51fe-2acf-4f05-b8f7-b849134bbfd4
            text = example.get("text", "") or ""
            sequences.append(text)
        #     sequences.append(text) if text != "" else None
        # assert len(sequences) >= 1, f'No sequences found in {data}'  # perhaps we do want to train on empty strings?

        # - Tokenize the sequences
        # tokenized_data = tokenizer(sequences, padding='longest', truncation=True, return_tensors='pt')  # TODO: we should probably set the max_length see discussion: https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76929999#76929999       # TODO: curious, how does the context length of model interact with this, will it be truncated by the HF model later if it's too long?
        # tokenized_data = tokenizer(sequences["text"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")  
        tokenized_data = tokenizer(sequences, padding="max_length", max_length=context_length, truncation=True, return_tensors="pt")  
        tokenized_data["labels"] = tokenized_data["input_ids"].clone()  # labels is hardcoded in HF so put it!
        return tokenized_data

    # - Debug before trianing to see data
    sample_data = train_dataset.select(range(per_device_train_batch_size)) if not isinstance(train_dataset, datasets.iterable_dataset.IterableDataset) else train_dataset.take(per_device_train_batch_size)
    processed_data = custom_collate_fn(sample_data, tokenizer=tokenizer)
    print(f'{processed_data=}')

    # -- Training arguments and trainer instantiation ref: https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        output_dir=Path('~/data/maf/results').expanduser(),  #The output directory where the model predictions and checkpoints will be written.
        # num_train_epochs = num_train_epochs, 
        max_steps=max_steps,  # TODO: hard to fix, see above
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=2,  # allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
        gradient_checkpointing = False,  # TODO depending on hardware set to true?
        optim="paged_adamw_32bit",  # David hall says to keep 32bit opt https://arxiv.org/pdf/2112.11446.pdf TODO: if we are using brain float 16 bf16 should we be using 32 bit? are optimizers always fb32?  https://discuss.huggingface.co/t/is-there-a-paged-adamw-16bf-opim-option/51284
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=500,  # TODO: once real training starts we can select this number for llama v2, what does llama v2 do to make it stable while v1 didn't?
        # warmup_ratio=0.03,  # number of steps for a linear warmup, TODO once real training starts change? 
        weight_decay=0.01,  # TODO once real training change?
        learning_rate = 5e-5,  # TODO once real training change? anything larger than -3 I've had terrible experiences with
        max_grad_norm=0.5, # TODO once real training change?
        lr_scheduler_type="cosine",  # TODO once real training change? using what I've seen most in vision 
        logging_dir=Path('~/data/maf/logs').expanduser(),
        save_steps=500,
        logging_steps=500,
        remove_unused_columns=False,  # TODO don't get why https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76929999#76929999
        report_to='none',  # change to wandb!
        fp16=False,  # never ever set to True
        bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8,  # if >= 8 ==> brain float 16 available or set to True if you always want fp32
    )

    # TODO: might be nice to figure our how llamav2 counts the number of token's they've trained on
    trainer = Trainer(
        model=model,
        args=training_args,  
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda data: custom_collate_fn(data, tokenizer=tokenizer)
    )
    # - TODO bellow is for qlora from falcon, has same interface as Trainer later lets use: https://github.com/artidoro/qlora
    # from trl import SFTTrainer
    # peft_config = None
    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=trainset,
    #     peft_config=peft_config,
    #     dataset_text_field="text",
    #     max_seq_length=max_seq_length,
    #     tokenizer=tokenizer,
    #     args=training_arguments,
    # )
    # TODO why this? https://discuss.huggingface.co/t/why-do-you-need-to-re-upcast-the-norm-layers-of-hf-falcon-to-fb32/46139
    # for name, module in trainer.model.named_modules():
    #     if "norm" in name:
    #         module = module.to(torch.float32)

    # - Train
    trainer.train()
    print('Done!\a')


# -- Run __main__

if __name__ == '__main__':
    print(f'\n\n\n------------------- Running {__file__} -------------------')
    # -- Run tests and time it
    import time
    time_start = time.time()
    # -- Run tests
    train()
    # -- End tests, report how long it took in seconds, minutes, hours, days
    print(f'Time it took to run {__file__}: {time.time() - time_start} seconds, {(time.time() - time_start)/60} minutes, {(time.time() - time_start)/60/60} hours, {(time.time() - time_start)/60/60/24} days\a')
