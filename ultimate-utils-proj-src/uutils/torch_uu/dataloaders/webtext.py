from argparse import Namespace
from pathlib import Path
from typing import Optional, Callable

import torch
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, Compose

import os
import pickle
from PIL import Image
import numpy as np
import tiktoken
import random

import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
from torch.utils.data import IterableDataset

from datasets import load_dataset, DatasetDict

from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders

class GPTIterableDataset(IterableDataset):
    def __init__(self, dataset, block_size):
        self.ids = dataset
        self.block_size = block_size

    def __iter__(self):
        while True:
            # sample a starting index for the example
            idx = random.randint(0, len(self.ids) - self.block_size - 2)
            # return the next block_size ids as input
            x = torch.from_numpy((self.ids[idx : idx + self.block_size]).astype(np.int64))
            # return the next indices as output
            y = torch.from_numpy((self.ids[idx + 1 : idx + 1 + self.block_size]).astype(np.int64))
            yield x, y

def get_dataloaders_for_webtext(args):
    # total_dataset = load_dataset('stas/openwebtext-10k', split = 'train')

    split_dataset = DatasetDict()
    split_dataset['train'] = load_dataset('stas/openwebtext-10k', split = 'train[:20%]')
    split_dataset['test'] = load_dataset('stas/openwebtext-10k', split = 'train[20%:25%]')
    split_dataset['val'] = load_dataset('stas/openwebtext-10k', split = 'train[25%:30%]')

    # split_dataset['train'], split_dataset['test'], split_dataset['val'] = random_split(total_dataset, [8000, 1000, 1000])

    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # print(split_dataset)
    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        # num_proc=2, # TODO
    )
    # print(tokenized)

    # combine all examples in each dataset into a single list of ids
    train_dataset = np.concatenate(tokenized['train']['ids'])
    test_dataset = np.concatenate(tokenized['test']['ids'])
    val_dataset = np.concatenate(tokenized['val']['ids'])

    train_loader = DataLoader(GPTIterableDataset(train_dataset, args.model_hps['block_size']), args.batch_size)
    test_loader = DataLoader(GPTIterableDataset(test_dataset, args.model_hps['block_size']), args.batch_size_eval)
    val_loader = DataLoader(GPTIterableDataset(val_dataset, args.model_hps['block_size']), args.batch_size_eval)




    # print(train_dataset)

    # need to create dataloaders for each dataset that generate random samples

    # # concatenate all the ids in each dataset into one large file we can use for training
    # for split, dset in tokenized.items():
    #     arr_len = np.sum(dset['len'])
    #     # filename = f'{split}.bin'
    #     # dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    #     # arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    #     print(f"writing {filename}...")
    #     idx = 0
    #     for example in tqdm(dset):
    #         arr[idx : idx + example['len']] = example['ids']
    #         idx += example['len']
    #     arr.flush()

    # train_file_loc = os.join(args.data_root, 'train.pkl')

    # if os.path.exists(train_file_loc):
    #     with open(train_file_loc) as f:
    #         train_dataset = pickle.load(f)
    # else:
    #     print("downloading train split")
    #     train_dataset = datasets.load_dataset('openwebtext', split = 'train')
    #     with open(train_file_loc, "wb") as f:
    #         pickle.dump(train_dataset, f)


    # test_file_loc = os.join(args.data_root, 'test.pkl')

    # if os.path.exists(test_file_loc):
    #     with open(test_file_loc) as f:
    #         test_dataset = pickle.load(f)
    # else:
    #     print("downloading test split")
    #     test_dataset = datasets.load_dataset('openwebtext', split = 'test')
    #     with open(test_file_loc, "wb") as f:
    #         pickle.dump(test_dataset, f)


    # val_file_loc = os.join(args.data_root, 'val.pkl')

    # if os.path.exists(val_file_loc):
    #     with open(val_file_loc) as f:
    #         val_dataset = pickle.load(f)
    # else:
    #     print("downloading val split")
    #     val_dataset = datasets.load_dataset('openwebtext', split = 'val')
    #     with open(val_file_loc, "wb") as f:
    #         pickle.dump(val_dataset, f)


    # train_loader, val_loader = get_serial_or_distributed_dataloaders(
    #     train_dataset=tokenized['train'],
    #     val_dataset=tokenized['val'],
    #     batch_size=args.batch_size,
    #     batch_size_eval=args.batch_size_eval,
    #     rank=args.rank,
    #     world_size=args.world_size
    # )

    # _, test_loader = get_serial_or_distributed_dataloaders(
    #     train_dataset=tokenized['test'],
    #     val_dataset=tokenized['test'],
    #     batch_size=args.batch_size,
    #     batch_size_eval=args.batch_size_eval,
    #     rank=args.rank,
    #     world_size=args.world_size
    # )

    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dataloaders

