"""

python -m torch.distributed.launch --nproc_per_node 2 main_data_parallel_ddp_pg.py
torchrun --nnodes 2 main_data_parallel_ddp_pg.py


- ref: https://discuss.huggingface.co/t/how-to-run-an-end-to-end-example-of-distributed-data-parallel-with-hugging-faces-trainer-api-ideally-on-a-single-node-multiple-gpus/21750/3


- SO: https://stackoverflow.com/questions/73391230/how-to-run-an-end-to-end-example-of-distributed-data-parallel-with-hugging-face
- training on multiple gpus: https://huggingface.co/docs/transformers/perf_train_gpu_many#efficient-training-on-multiple-gpus
- data paralelism, dp vs ddp: https://huggingface.co/docs/transformers/perf_train_gpu_many#data-parallelism
- github example: https://github.com/huggingface/transformers/tree/main/examples/pytorch#distributed-training-and-mixed-precision
    - above came from hf discuss: https://discuss.huggingface.co/t/using-transformers-with-distributeddataparallel-any-examples/10775/7

⇨ Single Node / Multi-GPU

Model fits onto a single GPU:

DDP - Distributed DP
ZeRO - may or may not be faster depending on the situation and configuration used.

...https://huggingface.co/docs/transformers/perf_train_gpu_many#scalability-strategy

python -m torch.distributed.launch \
    --nproc_per_node number_of_gpu_you_have path_to_script.py \
	--all_arguments_of_the_script

python -m torch.distributed.launch --nproc_per_node 2 ~/ultimate-utils/tutorials_for_myself/my_hf_hugging_face_pg/main_data_parallel_ddp_pg.py

e.g.
python -m torch.distributed.launch \
    --nproc_per_node 8 pytorch/text-classification/run_glue.py \

    --model_name_or_path bert-large-uncased-whole-word-masking \
    --task_name mnli \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/mnli_output/
"""
# %%

# - init group
# - set up processes a la l2l
# local_rank: int = local_rank: int = int(os.environ["LOCAL_RANK"]) # get_local_rank()
# print(f'{local_rank=}')
## init_process_group_l2l(args, local_rank=local_rank, world_size=args.world_size, init_method=args.init_method)
# init_process_group_l2l bellow
# if is_running_parallel(rank):
#     print(f'----> setting up rank={rank} (with world_size={world_size})')
#     # MASTER_ADDR = 'localhost'
#     MASTER_ADDR = '127.0.0.1'
#     MASTER_PORT = master_port
#     # set up the master's ip address so this child process can coordinate
#     os.environ['MASTER_ADDR'] = MASTER_ADDR
#     print(f"---> {MASTER_ADDR=}")
#     os.environ['MASTER_PORT'] = MASTER_PORT
#     print(f"---> {MASTER_PORT=}")
#
#     # - use NCCL if you are using gpus: https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
#     if torch.cuda.is_available():
#         backend = 'nccl'
#         # You need to call torch_uu.cuda.set_device(rank) before init_process_group is called. https://github.com/pytorch/pytorch/issues/54550
#         torch.cuda.set_device(
#             args.device)  # is this right if we do parallel cpu? # You need to call torch_uu.cuda.set_device(rank) before init_process_group is called. https://github.com/pytorch/pytorch/issues/54550
#     print(f'---> {backend=}')
# rank: int = torch.distributed.get_rank() if is_running_parallel(local_rank) else -1

# https://huggingface.co/docs/transformers/tasks/translation
import datasets
from datasets import load_dataset, DatasetDict

books: DatasetDict = load_dataset("opus_books", "en-fr")
print(f'{books=}')

books: DatasetDict = books["train"].train_test_split(test_size=0.2)
print(f'{books=}')
print(f'{books["train"]=}')

print(books["train"][0])
"""
{'id': '90560',
 'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
  'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}
"""

# - t5 tokenizer

from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("t5-small")
print(f'{isinstance(tokenizer, PreTrainedTokenizer)=}')
print(f'{isinstance(tokenizer, PreTrainedTokenizerFast)=}')

source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Then create a smaller subset of the dataset as previously shown to speed up the fine-tuning: (hack to seep up tutorial)
books['train'] = books["train"].shuffle(seed=42).select(range(100))
books['test'] = books["test"].shuffle(seed=42).select(range(100))

# # use 🤗 Datasets map method to apply a preprocessing function over the entire dataset:
# tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=2)

# todo - would be nice to remove this since gpt-2/3 size you can't preprocess the entire data set...or can you?
# tokenized_books = books.map(preprocess_function, batched=True, batch_size=2)
from uutils.torch_uu.data_uu.hf_uu_data_preprocessing import preprocess_function_translation_tutorial

preprocessor = lambda examples: preprocess_function_translation_tutorial(examples, tokenizer)
tokenized_books = books.map(preprocessor, batched=True, batch_size=2)
print(f'{tokenized_books=}')

# - load model
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# - to DDP
# model = model().to(rank)
# from torch.nn.parallel import DistributedDataParallel as DDP
# ddp_model = DDP(model, device_ids=[rank])

# Use DataCollatorForSeq2Seq to create a batch of examples. It will also dynamically pad your text and labels to the
# length of the longest element in its batch, so they are a uniform length.
# While it is possible to pad your text in the tokenizer function by setting padding=True, dynamic padding is more efficient.

from transformers import DataCollatorForSeq2Seq

# Data collator that will dynamically pad the inputs received, as well as the labels.
data_collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

"""
At this point, only three steps remain:

- Define your training hyperparameters in Seq2SeqTrainingArguments.
- Pass the training arguments to Seq2SeqTrainer along with the model, dataset, tokenizer, and data collator.
- Call train() to fine-tune your model.
"""
report_to = "none"
if report_to != 'none':
    import wandb
    wandb.init(project="playground", entity="brando", name='run_name', group='expt_name')

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# fp16 = True # cuda
# fp16 = False # cpu
import torch

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
    report_to=report_to,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

print('\n ----- Success\a')
