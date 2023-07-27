# Is Pre-training Truly Better Than Meta-Learning?: Tutorial on Computing Task2Vec Diversity of a Few-Shot Dataset

Our paper **Is Pre-training Truly Better Than Meta-Learning?** (https://arxiv.org/pdf/2306.13841.pdf) highlights how the Task2Vec diversity coefficient is a driving factor behind the relative performance between PT (pre-training) and MAML (meta-learning).  
This tutorial will demonstrate how to compute the Task2Vec diversity of a given few-shot dataset.

# Prerequisites
First, we need to install ultimate-utils and its prerequisites - Brando has created an excellent guide on how to do so here: https://github.com/brando90/ultimate-utils#readme

# Diversity Computation
This example computes the Task2Vec diversity of the training split of the FGVC-Aircraft (https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) benchmark:
```python
from uutils.argparse_uu.meta_learning import parse_args_meta_learning, fix_for_backwards_compatibility
from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
from uutils.torch_uu.metrics.diversity.diversity import \
    get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks
from uutils.torch_uu.metrics.diversity.task2vec_based_metrics import task_similarity
from uutils.torch_uu.models.probe_networks import get_probe_network
from uutils.torch_uu.metrics.diversity.diversity import get_task_embeddings_from_few_shot_l2l_benchmark
from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility

# default args for meta-learning or n-way k-shot few-shot benchmarks
args = parse_args_meta_learning()

# args for computing task2vec diversity on aircraft via a resnet18_pretrained probe
args.data_path = '~/data/l2l_data' # or whereever you want your datasets (in this case aircraft) installed
args.batch_size = 2
args.batch_size_eval = args.batch_size  # this determines batch size for test/eval
args.data_option = 'aircraft'
args.data_augmentation = 'hdb4_micod'
args.model_option = 'resnet18_pretrained_imagenet' # - probe_network
args.classifier_opts = None

args = fix_for_backwards_compatibility(args)

args.probe_network = get_probe_network(args) # - create probe_network
args.tasksets = get_l2l_tasksets(args) # - create taskloader

split = 'train' # 'train' or 'validation' or 'test'

# get list of task2vec embeddings for our dataset
embeddings = get_task_embeddings_from_few_shot_l2l_benchmark(args.tasksets,
                                                            args.probe_network,
                                                            split=split,
                                                            num_tasks_to_consider=args.batch_size,
                                                            classifier_opts=args.classifier_opts,
                                                            )

# - compute distance matrix & task2vec based diversity, this code computes pair-wise distance between task embeddings
distance_matrix = task_similarity.pdist(embeddings, distance='cosine')

# - compute 95% confidence interval of div
div, ci = get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks(distance_matrix)
print(f'Diversity: {(div, ci)=}')
```

