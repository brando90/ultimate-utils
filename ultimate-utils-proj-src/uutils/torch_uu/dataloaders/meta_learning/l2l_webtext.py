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

def get_dataloaders_for_webtext(args):
    # total_dataset = load_dataset('stas/openwebtext-10k', split = 'train')

    tokenized_dataset = get_tokenized_webtext()
    train_loader = DataLoader(GPTIterableDataset(tokenized_dataset, block_size = args.model_hps['block_size'], split = 'train'), args.batch_size)
    val_loader = DataLoader(GPTIterableDataset(tokenized_dataset, block_size = args.model_hps['block_size'], split = 'val'), args.batch_size)

    # do not need test dataset
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': val_loader}
    return dataloaders

