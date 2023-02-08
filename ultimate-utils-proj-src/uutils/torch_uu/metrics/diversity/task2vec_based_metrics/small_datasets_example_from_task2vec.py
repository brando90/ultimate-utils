# %%
"""
https://colab.research.google.com/drive/1DoIbWPmfuIaNul2eP4IYhEn23HfHW5zJ#scrollTo=UXuWoGkvNZ8t
"""
# !pip install -U git+https://github.com/brando90/ultimate-aws-cv-task2vec
# pip install -U git+https://github.com/moskomule/anatome

# %%
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import task2vec
from task2vec import Task2Vec
from models import get_model
import datasets
import task_similarity

# %%

# dataset_names = ('stl10', 'mnist', 'cifar10', 'cifar100', 'letters', 'kmnist')
# dataset_names = ('mnist', 'cifar10')
dataset_names = ('mnist', )
# dataset_names = ('cifar100', 'letters')
# dataset_names = ('letters', 'cifar100')
# Change `root` with the directory you want to use to download the datasets
dataset_list = [datasets.__dict__[name](root=Path('/').expanduser())[0] for name in dataset_names]

# %%

print('---- Combing task2vec embeddings ----')
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

embeddings = []
for name, dataset in zip(dataset_names, dataset_list):
    print(f"-- {name=}")
    probe_network = get_model('resnet18', pretrained=True, num_classes=int(max(dataset.targets) + 1)).to(device)
    print(f'{probe_network=}')
    # embeddings.append(Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(dataset)).to(device)
    # embeddings.append(Task2Vec(probe_network, max_samples=100, skip_layers=6).embed(dataset))
    embedding: task2vec.Embedding = Task2Vec(deepcopy(probe_network)).embed(dataset)
    embeddings.append(embedding)
    print()

print(f'{embeddings=}')
# %%

distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
print(f'{distance_matrix=}')
# %%

task_similarity.plot_distance_matrix(embeddings, dataset_names)
