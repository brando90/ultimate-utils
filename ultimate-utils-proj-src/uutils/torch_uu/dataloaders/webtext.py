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
from uutils.torch_uu.dataset.webtext_tokenizer import get_tokenized_webtext

import learn2learn as l2l
from learn2learn.vision.benchmarks import BenchmarkTasksets

class GPTIterableDataset(IterableDataset):
    def __init__(self, tokenized_dataset, block_size, split = 'train'):
        """
        Get IterableDataset for tokenized OpenWebText
        tokenized_dataset: np.memmap object
        """
        self.ids = tokenized_dataset
        self.dataset_len = tokenized_dataset.size

        if split == 'train':
            self.split_start = 0
            self.split_end = int(self.dataset_len*0.999)
        else:
            self.split_start = int(self.dataset_len*0.999)
            self.split_end = self.dataset_len

        self.split_len = self.split_end - self.split_start
        self.block_size = block_size

    def __iter__(self):
        while True:
            # sample a starting index for the example
            idx = random.randint(0, self.split_len - self.block_size - 2)
            # return the next block_size ids as input
            x = torch.from_numpy((self.ids[idx + self.split_start : idx + self.split_start + self.block_size]).astype(np.int64))
            # return the next indices as output
            y = torch.from_numpy((self.ids[idx + self.split_start + 1 : idx + self.split_start + 1 + self.block_size]).astype(np.int64))
            yield x, y


class GPT_L2L_Dataset(Dataset):
    def __init__(self, tokenized_dataset, block_size, vocab_size, split = 'train'):
        """
        Get Dataset for L2L on tokenized OpenWebText
        tokenized_dataset: np.memmap object
        """
        self.ids = tokenized_dataset
        self.dataset_len = tokenized_dataset.size
        self.labels = list(range(vocab_size))

        if split == 'train':
            self.split_start = 0
            self.split_end = int(self.dataset_len*0.999)
        else:
            self.split_start = int(self.dataset_len*0.999)
            self.split_end = self.dataset_len

        self.split_len = self.split_end - self.split_start
        self.block_size = block_size

        print("l2l size:", self.split_len - self.block_size)

    def __getitem__(self, idx):
        # start_idx = idx//self.block_size
        # num_tokens = (idx % self.block_size) + 1
        # return the next block_size ids as input
        x = torch.from_numpy((self.ids[idx + self.split_start : idx + self.split_start + self.block_size]).astype(np.int64))
        # return the next indices as output
        y = torch.from_numpy((self.ids[idx + self.split_start + self.block_size:idx + self.split_start + self.block_size + 1]).astype(np.int64))
        return x, y

    def __len__(self):
        # # for each starting index, can give an input with between 1 and self.block_size number of tokens
        # return (self.split_len - self.block_size)*self.block_size

        # for now, assume all examples are of length block_size
        return self.split_len - self.block_size



def get_l2l_tasksets_for_webtext(args):
    return _get_l2l_tasksets_for_webtext(args.model_hps['block_size'], args.model_hps['vocab_size'])

def _get_l2l_tasksets_for_webtext(block_size, vocab_size):
    pkl_dir = '/lfs/hyperion/0/saumg/'
    benchmark_tasksets_file = pkl_dir+'openwebtext_benchmarktasksets.pkl'
    if os.path.exists(benchmark_tasksets_file):
        with open(benchmark_tasksets_file, 'rb') as f:
            return pickle.load(f)

    else:
        tokenized_dataset = get_tokenized_webtext()
        ## WARNING: change below val to train
        train_taskdataset_file = pkl_dir+'openwebtext_val_taskdataset.pkl'
        if os.path.exists(train_taskdataset_file):
            with open(train_taskdataset_file, 'rb') as f:
                train_tasks = pickle.load(f)
        else:
            train_dataset = GPT_L2L_Dataset(tokenized_dataset, block_size = block_size, vocab_size = vocab_size, split = 'train')
            print("starting creation of TaskDataset for train...")
            train_meta = l2l.data.MetaDataset(train_dataset)
            train_tasks = l2l.data.TaskDataset(dataset=train_meta, task_transforms = [l2l.data.transforms.FusedNWaysKShots(val_tasks, n = 4, k = 1)])
            print("done, dumping")
            with open(train_taskdataset_file, 'wb+') as f:
                pickle.dump(train_tasks, f)
            print("dumped")

        val_taskdataset_file = pkl_dir+'openwebtext_val_taskdataset.pkl'
        if os.path.exists(val_taskdataset_file):
            with open(val_taskdataset_file,'rb') as f:
                val_tasks = pickle.load(f)
        else:
            val_dataset = GPT_L2L_Dataset(tokenized_dataset, block_size = block_size, vocab_size = vocab_size, split = 'val')
            print("starting creation of TaskDataset for val...")
            val_meta = l2l.data.MetaDataset(val_dataset)
            val_tasks = l2l.data.TaskDataset(dataset=val_meta, task_transforms = [l2l.data.transforms.FusedNWaysKShots(val_tasks, n = 4, k = 1)])
            print("done, dumping")
            with open(val_taskdataset_file, 'wb+') as f:
                pickle.dump(val_tasks, f)
            print("dumped")

        print("starting creation of BenchmarkTasksets...")
        benchmark_tasksets = BenchmarkTasksets(train_tasks, val_tasks, val_tasks)
        print("done, dumping")
        with open(benchmark_tasksets_file, 'wb+') as f:
            pickle.dump(benchmark_tasksets, f)
        print("dumped")

        return benchmark_tasksets

        

    # tokenized_dataset = get_tokenized_webtext()
    # train_dataset = GPT_L2L_Dataset(tokenized_dataset, block_size = args.model_hps['block_size'], split = 'train')
    # val_dataset = GPT_L2L_Dataset(tokenized_dataset, block_size = args.model_hps['block_size'], split = 'val')

    # train_tasks = l2l.data.TaskDataset(dataset=train_dataset)
    # test_tasks = l2l.data.TaskDataset(dataset = val_dataset)
    # val_tasks = l2l.data.TaskDataset(dataset=val_dataset)

    # return BenchmarkTasksets(train_tasks, validation_tasks, test_tasks)



def get_dataloaders_for_webtext(args):
    # total_dataset = load_dataset('stas/openwebtext-10k', split = 'train')

    tokenized_dataset = get_tokenized_webtext()
    train_loader = DataLoader(GPTIterableDataset(tokenized_dataset, block_size = args.model_hps['block_size'], split = 'train'), args.batch_size)
    val_loader = DataLoader(GPTIterableDataset(tokenized_dataset, block_size = args.model_hps['block_size'], split = 'val'), args.batch_size)

    # do not need test dataset
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': val_loader}
    return dataloaders

    # split_dataset = DatasetDict()
    # # split_dataset['train'] = load_dataset('stas/openwebtext-10k', split = 'train[:20%]', cache_dir = '/lfs/hyperion/0/saumg/')
    # # split_dataset['test'] = load_dataset('stas/openwebtext-10k', split = 'train[20%:25%]', cache_dir = '/lfs/hyperion/0/saumg/')
    # # split_dataset['val'] = load_dataset('stas/openwebtext-10k', split = 'train[25%:30%]', cache_dir = '/lfs/hyperion/0/saumg/')

    # split_name = 'train'
    # tokenized_file_loc = '/lfs/hyperion/0/saumg/openwebtext_'+split_name+'_tokenised.pkl'


    # if not os.path.exists(tokenized_file_loc):
    #     total_dataset = DatasetDict()
    #     total_dataset['train'] = load_dataset('openwebtext', split = split_name, cache_dir = '/lfs/hyperion/0/saumg/')

        
    #     # # sanity check to remove later
    #     # total_len = len(total_dataset)
    #     # split_dataset['train'], split_dataset['test'], split_dataset['val'] = random_split(total_dataset, [total_len*0.8, total_len*0.1, total_len*0.1])
        

    #     enc = tiktoken.get_encoding("gpt2")
    #     def process(example):
    #         ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    #         ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    #         # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    #         out = {'ids': ids, 'len': len(ids)}
    #         return out

    #     # print(split_dataset)
    #     # tokenize the dataset
    #     tokenized = total_dataset.map(
    #         process,
    #         remove_columns=['text'],
    #         desc="tokenizing the splits",
    #         # num_proc=8, # TODO
    #     )

    #     tokenized_dataset = np.concatenate(tokenized['train']['ids'])
    #     print("storing tokenized dataset file...")
    #     with open(tokenized_file_loc, 'w+') as fil:
    #         pickle.dump(tokenized_dataset, fil)

    # else:
    #     print("opening tokenized dataset file...")
    #     with open(tokenized_file_loc, 'r') as fil:
    #         tokenized_dataset = pickle.load(fil)



    # # split_dataset['train'] = load_dataset('openwebtext', split = 'train[:5%]', cache_dir = '/lfs/hyperion/0/saumg/')
    # # split_dataset['test'] = load_dataset('openwebtext', split = 'train[5%:6%]', cache_dir = '/lfs/hyperion/0/saumg/')
    # # split_dataset['val'] = load_dataset('openwebtext', split = 'train[6%:7%]', cache_dir = '/lfs/hyperion/0/saumg/')

    # # # split_dataset['train'], split_dataset['test'], split_dataset['val'] = random_split(total_dataset, [8000, 1000, 1000])

    # # enc = tiktoken.get_encoding("gpt2")
    # # def process(example):
    # #     ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    # #     ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # #     # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    # #     out = {'ids': ids, 'len': len(ids)}
    # #     return out

    # # # print(split_dataset)
    # # # tokenize the dataset
    # # tokenized = split_dataset.map(
    # #     process,
    # #     remove_columns=['text'],
    # #     desc="tokenizing the splits",
    # #     # num_proc=2, # TODO
    # # )
    # # # print(tokenized)

    # # # combine all examples in each dataset into a single list of ids
    # # train_dataset = np.concatenate(tokenized['train']['ids'])
    # # test_dataset = np.concatenate(tokenized['test']['ids'])
    # # val_dataset = np.concatenate(tokenized['val']['ids'])

    # dataset_len = len(tokenized_dataset)
    # train_dataset = tokenized_dataset[:0.8*dataset_len]
    # test_dataset = tokenized_dataset[0.8*dataset_len:0.9*dataset_len]
    # val_dataset = tokenized_dataset[0.9*dataset_len:dataset_len]

    # train_loader = DataLoader(GPTIterableDataset(train_dataset, args.model_hps['block_size']), args.batch_size)
    # test_loader = DataLoader(GPTIterableDataset(test_dataset, args.model_hps['block_size']), args.batch_size_eval)
    # val_loader = DataLoader(GPTIterableDataset(val_dataset, args.model_hps['block_size']), args.batch_size_eval)




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

    # dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # return dataloaders

