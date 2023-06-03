# Computing metrics & tutorials

## Computing Diversity tutorial

For background on our formal diversity we recommend our short [6 page reference](https://brando90.github.io/brandomiranda/professional_documents/ICML_2023_DeployGenAI_Workshop__Diversity_Coefficient___LLMs__6pg_.pdf and longer arxiv link (to be released soon).
For more references see the end of this document or see the related work section of the paper previously referenced. 

### Compute Diversity using normal pytorch dataloader

```python
# -- lets create som dummy data set
import torch
import random


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, index):
        data = torch.randn(1, 1)  # generate random data
        label = random.randint(0, 1)  # generate random label
        return data, label

    def __len__(self):
        return self.num_samples


# - create minimal an args object needed for code to work
from argparse import Namespace

args = Namespace(batch_size=32)
args.classifier_opts = None
dataset = RandomDataset(100)  # 100 samples
from torch import nn

args.probe_network = nn.Linear(1, 1)
split = 'train'

# - get your pytorch data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
args.dataloaders = {'train': dataloader, 'val': dataloader, 'test': dataloader}

# - get task embeddings from normal pytorch data loader
from uutils.torch_uu.metrics.diversity.diversity import get_task_embeddings_from_few_shot_dataloader
from uutils.torch_uu.metrics.diversity.task2vec_based_metrics import task2vec, task_similarity

embeddings: list[task2vec.Embedding] = get_task_embeddings_from_few_shot_dataloader(args,
                                                                                    args.dataloaders,
                                                                                    args.probe_network,
                                                                                    num_tasks_to_consider=args.batch_size,
                                                                                    split=split,
                                                                                    classifier_opts=args.classifier_opts,
                                                                                    )

# - compute distance matrix & task2vec based diversity, to demo` task2vec, this code computes pair-wise distance between task embeddings
import numpy as np

distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
print(f'{distance_matrix=}')
from uutils.numpy_uu.common import get_diagonal

distances_as_flat_array, _, _ = get_diagonal(distance_matrix, check_if_symmetric=True)

# - compute div
from uutils.torch_uu.metrics.diversity.diversity import

get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks,
get_standardized_diversity_coffecient_from_pair_wise_comparison_of_tasks

div_tot = float(distances_as_flat_array.sum())
print(f'Diversity: {div_tot=}')
div, ci = get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks(distance_matrix)
print(f'Diversity: {(div, ci)=}')
standardized_div: float = get_standardized_diversity_coffecient_from_pair_wise_comparison_of_tasks(distance_matrix)
print(f'Standardised Diversity: {standardized_div=}')
```

### Compute complexity using normal pytorch dataloader

```python
    # ... see above code

# - get task embeddings from normal pytorch data loader
embeddings: list[task2vec.Embedding] = get_task_embeddings_from_few_shot_dataloader(args,
                                                                                    args.dataloaders,
                                                                                    args.probe_network,
                                                                                    num_tasks_to_consider=args.batch_size,
                                                                                    split=split,
                                                                                    classifier_opts=args.classifier_opts,
                                                                                    )

# - compute distance matrix & task2vec based diversity, to demo` task2vec, this code computes pair-wise distance between task embeddings
distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
print(f'{distance_matrix=}')
distances_as_flat_array, _, _ = get_diagonal(distance_matrix, check_if_symmetric=True)

# - compute complexity of benchmark (p determines which L_p norm we use to compute complexity. See task2vec_norm_complexity.py for details)
from uutils.torch_uu.metrics.complexity.task2vec_norm_complexity import standardized_norm_complexity  # pycharm bug?

p_norm = 1  # Set 1 for L1 norm, 2 for L2 norm, etc. 'nuc' for nuclear norm, np.inf for infinite norm
all_complexities = get_task_complexities(embeddings, p=p_norm)
print(f'{all_complexities=}')
complexity_tot = total_norm_complexity(all_complexities)
print(f'Total Complexity: {complexity_tot=}')
complexity_avg, complexity_ci = avg_norm_complexity(all_complexities)
print(f'Average Complexity: {(complexity_avg, complexity_ci)=}')
standardized_norm_complexity = standardized_norm_complexity(embeddings)
print(f'Standardized Norm Complexity: {standardized_norm_complexity=}')
```

### Compute diversity using an l2l dataloader

url for upcoming paper: https://github.com/brando90/diversity-for-predictive-success-of-meta-learning/blob/main/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py

```python
# - args for l2l
from pathlib import Path
from argparse import Namespace
args: Namespace = Namespace()
args.batch_size = 500
args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
args.data_path = Path('~/data/l2l_data/').expanduser()
args.data_augmentation = 'lee2019'
split = 'train'

# - get task embeddings from l2l dataloader
from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
from learn2learn.vision.benchmarks import BenchmarkTasksets
args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)
print(f'{args.tasksets=}')
from uutils.torch_uu.metrics.diversity.diversity import get_task_embeddings_from_few_shot_l2l_benchmark
from uutils.torch_uu.metrics.diversity.task2vec_based_metrics import task2vec, task_similarity
embeddings: list[task2vec.Embedding] = get_task_embeddings_from_few_shot_l2l_benchmark(args.tasksets,
                                                                                       args.probe_network,
                                                                                       split=split,
                                                                                       num_tasks_to_consider=args.batch_size,
                                                                                       )
        # - # - compute div (same as code above, you only need list of embeddings!)
```

## Citations

If you use this implementation please consider citing us:

```
@article{Miranda2022,
   author = {Brando Miranda and Patrick Yu and Yu-Xiong Wang and Sanmi Koyejo},
   doi = {10.48550/arxiv.2208.01545},
   month = {8},
   title = {The Curse of Low Task Diversity: On the Failure of Transfer Learning to Outperform MAML and Their Empirical Equivalence},
   url = {https://arxiv.org/abs/2208.01545v1},
   year = {2022},
}
```

arxiv url: https://arxiv.org/abs/2208.01545.

If you use our implementations for diversities for Large Language Models (LLMs) or Foundation Models (FMs)
consider citing us:

```
# todo: change low div to LLMs div later.

@article{Miranda2022,
   author = {Brando Miranda and Patrick Yu and Yu-Xiong Wang and Sanmi Koyejo},
   doi = {10.48550/arxiv.2208.01545},
   month = {8},
   title = {The Curse of Low Task Diversity: On the Failure of Transfer Learning to Outperform MAML and Their Empirical Equivalence},
   url = {https://arxiv.org/abs/2208.01545v1},
   year = {2022},
```

arxiv url: coming soon.

If you use our task2vec diversities or complexities consider citing the authors of Task2Vec:

```
@report{AchilleUCLA,
   author = {Alessandro Achille UCLA and Michael Lam AWS and Rahul Tewari AWS and Avinash Ravichandran AWS and Subhransu Maji UMass and Stefano Soatto UCLA and Pietro Perona Caltech},
   note = {task2vec<br/>d_sym(F_a, F_b) = d_cos(F_a/(F_a+F_b), F_b/(F_a+F_b))<br/><br/>task_embedding = diagonal of the FIM for the filters of size [F_total, 1] total filters for a network.<br/><br/>F_a = diagonal of the fisher information computed on the prob net.<br/>Division is element-wise},
   title = {TASK2VEC: Task Embedding for Meta-Learning Charless Fowlkes UCI and AWS},
   year = {2019},
}
```
