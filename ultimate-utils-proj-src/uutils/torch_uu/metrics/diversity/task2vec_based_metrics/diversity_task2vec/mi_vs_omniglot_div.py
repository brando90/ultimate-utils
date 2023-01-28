"""
Todo
- seperate the data augmentations from getting data
- assert the type & put return type in the function return
- get the data witht the right hdb1 data aug & next iter dl with a normal dl
- then try to use task2vec dl

"""
# dataset_list = [datasets.__dict__[name](root=Path('~/data').expanduser())[0] for name in dataset_names]
from datasets import Dataset
from pathlib import Path

import numpy as np
from copy import deepcopy

import torch

from diversity_src.diversity.task2vec_based_metrics.models import get_model

from diversity_src.diversity.task2vec_based_metrics import task2vec
from diversity_src.diversity.task2vec_based_metrics.task2vec import Task2Vec
from diversity_src.diversity.task2vec_based_metrics import task_similarity

from uutils import report_times, expanduser


# - test

def compute_mi_vs_omni_cosine_distance():
    """
    Let's estimate why the div is so low, is the distance btw omniglot and MI not high?
    I expect:
        stl10 vs letters = 0.6
        cifar100 vs mnist = 0.5
    """
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import get_mi_and_omniglot_list_data_set_splits

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    root: Path = expanduser('/l2l_data/')
    dataset_list_train, dataset_list_validation, dataset_list_test = get_mi_and_omniglot_list_data_set_splits(root,
                                                                                                              data_augmentation='hdb1',
                                                                                                              device=device)
    print('---- Computing task2vec embeddings ----')
    datasets = dataset_list_train
    dataset_names = [dataset.name for dataset in datasets]
    print(f'{datasets=}')
    print(f'{dataset_names=}')
    embeddings = []
    for dataset_name, dataset in zip(dataset_names, datasets):
        print(f"-- {dataset=}")
        print(f"-- {dataset_name=}")
        # assert isinstance(dataset, Dataset), f'{type(dataset)=}'
        num_classes = len(dataset.labels)
        # num_classes = int(max(dataset.targets) + 1)
        print(f'{num_classes=}')
        probe_network = get_model('resnet18', pretrained=True, num_classes=num_classes).to(device)
        print(f'{probe_network=}')
        # embeddings.append(Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(dataset)).to(device)
        # embeddings.append(Task2Vec(probe_network, max_samples=100, skip_layers=6).embed(dataset))
        embedding: task2vec.Embedding = Task2Vec(deepcopy(probe_network)).embed(dataset).to(device)
        embeddings.append(embedding)
        print()

    print(f'{embeddings=}')
    distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
    print(f'{distance_matrix=}')
    task_similarity.plot_distance_matrix(embeddings, dataset_names)


if __name__ == '__main__':
    import time
    from uutils import report_times, expanduser

    start = time.time()
    # - run experiment
    compute_mi_vs_omni_cosine_distance()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
