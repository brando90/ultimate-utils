# Is Pre-training Truly Better Than Meta-Learning?: Tutorial on Computing Task2Vec Diversity of a Few-Shot Dataset

Our paper **Is Pre-training Truly Better Than Meta-Learning?** (https://arxiv.org/pdf/2306.13841.pdf) highlights how the Task2Vec diversity coefficient is a driving factor behind the relative performance between PT (pre-training) and MAML (meta-learning).  
This tutorial will demonstrate how to compute the Task2Vec diversity of a given few-shot dataset.

# Prerequisites
First, we need to install ultimate-utils and its prerequisites - Brando has created an excellent guide on how to do so here: https://github.com/brando90/ultimate-utils#readme

# Tutorials Quick Start

## Tutorial 1 - Quick Start - Diversity Computation for few-shot learning vision datasets?
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
args.batch_size = 500 # how many n-way k-shot tasks you want to use to compute diversity. The larger this is, the tighter the confidence interval of your diversity
args.batch_size_eval = args.batch_size  
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

At the end of running the above code, you should get something like this (but likely with more decimal digits):
```                                                                                                               
Diversity: (div, ci)=(0.108, 0.003)
```
This means that our 95% confidence interval of the diversity coefficient of the FGVC-Aircraft dataset on the training split is [0.108-0.003, 0.108+0.003] or [0.105, 0.111].


# Additional Notes and Helpful Tips
At a high-level, the diversity computation example above works by 

1) computing a list of Task2Vec embeddings of our benchmark (stored in the variable `embeddings`),
2) computing the confidence interval of the expected distance between two unique Task2Vec embeddings sampled from this list (stored in `div, ci` and the confidence interval is [div-ci, div+ci]). 

As an intermediate step, the variable `distance_matrix` stores the pairwise cosine distances between any two Task2Vec embeddings.

To test a different dataset, you can replace `args.data_option` with another dataset, such as 'flower' for VGGFlower, 'dtd' for Describable Textures Dataset, 'omni' for Omniglot, 'hdbX' for a high-diversity benchmark (where X = {4_micod,6,7,8,9,10}).

You may also test a different Task2Vec probe network. The available probe networks are `resnet18_random, resnet18_pretrained, resnet34_random, resnet34_pretrained` where random denotes a probe network that is randomly initalized, and pretrained denotes a probe network that is pretrained on ImageNet. Resnet18 is a 18-layer deep residual network and Resnet34 is a 34-layer deep residual network.

Furthermore, you can play around with computing the Task2Vec diversity on different splits of your dataset of choice - you can set `split` to either train, validation, or test. 

Generally, the larger the batch size (`args.batch_size`), the tighter the confidence interval [div-ci, div+ci], and the more confident you are in your estimated diversity. Typically, a good value to set your batch size is `args.batch_size = 500` for a diversity result that is accurate to roughly three decimal places.
