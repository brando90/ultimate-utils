"""
File with code for computing diveristy:
    dv(B) = E_{tau1, tau2 ~ p(tau1, tau2 | B)}[ d(f1(tau1), f2(tau2) )]
the expected distance between tasks for a given benchmark B.

Example use:
1. Compute diveristy for one single few-shot learning benchmark:
    Example 1:
    - get 1 big (meta) batch of tasks  X e.g. [B, M, C, H, W]. Note X1=X1=X
    - get cross product of tasks we want to use to pair distance per tasks (to ultimately compute diveristy)
    - then for each pair of tasks compute distance for that task
    - then return distances as [N, L] for each layer according to base_models you gave - but the recommended layer to use
    to compute diveristy is L-1 (i.e. the feature extractor laye) and the the final L.
    Using the avg of all the layers in principle should work to but usually results in too high variance to be any use
    because each layer has very different distances as we get deeper in the network.
    - div = mean(distances_for_tasks_pairs[L-1]) or div = mean(distances_for_tasks_pairs[L]) or div = mean(distances_for_tasks_pairs[L-1, L])

    Example 2:
    - another option is to get two meta-batch of tasks and just feed them directly like in 2. Then one wouldn't have
    to worry as much wether to include the diagnoal or not.

2. Compute diveristy for pair of data sets/loaders:
    Example 1:
    - get 1 big batch of images for each data set/loader such that we have many classes (ideally all the classes for
    each data set)
    - treat a class as a task and construct two tensors X1, X2 of shapes [B1, M1, C1, H1, W1],  [B2, M2, C2, H2, W2]
    where B1, B2 are the number of images.
    - repeat what was done for example 1.1 from few-shot learning benchmark i.e. compute pair of pair distances of tasks
    and compute diversity from it.

3. Compute diversity using task2vec
    Example 1:
    - batch = get one very large batch
    - X = create [Concepts, M, C, H, W]
    - X1 = X2 = X
    - compute div with X
    Example 2 (SL data loaders):
    - batch1, batch2 = get two very large (normal) batches
    - X1, X2 = sort them such that the classes are sorted
    - pass them to distance matrix and compute div
"""
from argparse import Namespace
from collections import OrderedDict
import random
from pprint import pprint
from typing import Optional

import numpy as np
from torch import Tensor, nn

# from anatome.helper import LayerIdentifier, dist_data_set_per_layer, _dists_per_task_per_layer_to_list, compute_stats_from_distance_per_batch_of_data_sets_per_layer

from uutils.torch_uu import tensorify, process_meta_batch
from uutils.torch_uu.dataloaders.meta_learning.torchmeta_ml_dataloaders import get_miniimagenet_dataloaders_torchmeta, \
    get_minimum_args_for_torchmeta_mini_imagenet_dataloader
from uutils.torch_uu.metrics.diversity.task2vec_based_metrics import task2vec, task_similarity
from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner, \
    get_feature_extractor_conv_layers, get_last_two_layers

from pdb import set_trace as st


# get_distances_for_task_pair = dist_data_set_per_layer


def select_index(B: int, rand: bool = True):
    """
    Note:
        - probably could have been made an iterator with yield but decided to keep the code simple and not do it.
    :param B:
    :param rand:
    :return:
    """
    if rand:
        # [a, b], including both end points.
        i = random.randint(a=0, b=B - 1)
        return i
    else:
        raise ValueError(f'Not implemented with value: {rand=}')
        # yield i


def select_index_pair(B1, B2, rand: bool = True, consider_diagonal: bool = False) -> tuple[int, int]:
    """

    :param B1:
    :param B2:
    :param rand:
    :param consider_diagonal:
    :return:
    """
    while True:
        i1 = select_index(B1, rand)
        i2 = select_index(B2, rand)
        # - if no diagonal element then keep sampling until i1 != i2, note since there are N diagonals vs N^2 it
        # shouldn't take to long to select valid indices if we want random element with no diagonal.
        if not consider_diagonal:
            # if indicies are different then it's not a diagonal
            if i1 != i2:
                return i1, i2
        else:
            # - just return if we are ok with the diagonal e.g. when X1, X2 are likely very different at top call.
            return i1, i2


def get_list_tasks_tuples(B1: int, B2: int,
                          num_tasks_to_consider: int = 25,
                          rand: bool = True,
                          consider_diagonal: bool = False
                          ) -> list[tuple[int, int]]:
    """

    :param B1:
    :param B2:
    :param num_tasks_to_consider:
    :param rand:
    :param consider_diagonal:
    :return:
    """
    assert (num_tasks_to_consider <= B1 * B2)
    # - generate num_tasks_to_consider task index pairs
    list_task_index_pairs: list[tuple[int, int]] = []
    for i in range(num_tasks_to_consider):
        idx_task1, idx_task2 = select_index_pair(B1, B2, rand, consider_diagonal=consider_diagonal)
        list_task_index_pairs.append((idx_task1, idx_task2))
    assert len(list_task_index_pairs) == num_tasks_to_consider
    return list_task_index_pairs


# not recommended, legacy
def get_all_required_distances_for_pairs_of_tasks_using_cca_like_metrics(f1: nn.Module, f2: nn.Module,
                                                                         X1: Tensor, X2: Tensor,
                                                                         layer_names1: list[str],
                                                                         layer_names2: list[str],
                                                                         metric_comparison_type: str = 'pwcca',
                                                                         iters: int = 1,
                                                                         effective_neuron_type: str = 'filter',
                                                                         downsample_method: Optional[str] = None,
                                                                         downsample_size: Optional[int] = None,
                                                                         subsample_effective_num_data_method: Optional[
                                                                             str] = None,
                                                                         subsample_effective_num_data_param: Optional[
                                                                             int] = None,
                                                                         metric_as_sim_or_dist: str = 'dist',
                                                                         force_cpu: bool = False,

                                                                         num_tasks_to_consider: int = 25,
                                                                         consider_diagonal: bool = False
                                                                         ) -> list[OrderedDict[str, float]]:
    # ) -> list[OrderedDict[LayerIdentifier, float]]:
    """
    [L] x [B, n*k, C,H,W]^2 -> [L] x [B', n*k, C,H,W] -> [B', L]

    Compute the pairwise distances between the collection of tasks in X1 and X2:
        get_distances_for_task_pairs = [d(f(tau_s1), f(tau_s2))]_{s1,s2 \in {num_tasks_to_consider}}
    of size [B', L]. In particular notice how number of tasks used B' gives 1 distance value btw tasks (for a fixed layer).
    used to compute diversity:
        div = mean(get_distances_for_task_pairs)
        ci = ci(get_distances_for_task_pairs)

    Note:
        - make sure that M is safe i.e. M >= 10*D' e.g. for filter (image patch) based comparison
        D'= ceil(M*H*W) so M*H*W >= s*C for safety margin s say s=10.
        - if cls/head layer is taken into account and small (e.g. n_c <= D' for the other layers) then M >= s*n_c.

    :param f1:
    :param f2:
    :param X1: [B, M, D] or [B, M, C, H, W]
    :param X2: [B, M, D] or [B, M, C, H, W]
    :param layer_names1:
    :param layer_names2:
    :param num_tasks_to_consider: from the cross product of tasks in X1 and X2, how many to consider
        - num_tasks_to_consider = c*sqrt{B^2 -B} or c*B is a good number of c>=1.
        - num_tasks_to_consider >= 25 is likely a good number.
    :param consider_diagonal: if X1 == X2 then we do NOT want to consider the diagonal since that would compare the
    same tasks to each other and that has a distance = 0.0
    :return:
    """
    from anatome.helper import LayerIdentifier, dist_data_set_per_layer, _dists_per_task_per_layer_to_list, \
        compute_stats_from_distance_per_batch_of_data_sets_per_layer
    assert metric_as_sim_or_dist == 'dist'
    L = len(layer_names1)
    B1, B2 = X1.size(0), X2.size(0)
    B_ = num_tasks_to_consider
    assert B_ <= B1 * B2, f'You can\'t use more tasks than exist in the cross product of tasks, choose' \
                          f'{num_tasks_to_consider=} such that is less than or equal to {B1*B2=}.'
    assert L == len(layer_names1) == len(layer_names2)

    # - get indices for pair of tasks
    indices_task_tuples: list[tuple[int, int]] = get_list_tasks_tuples(B1, B2,
                                                                       num_tasks_to_consider=num_tasks_to_consider,
                                                                       rand=True,
                                                                       consider_diagonal=consider_diagonal
                                                                       )
    # [B']
    assert len(indices_task_tuples) == B_

    # - compute required distance pairs of tasks [B', L]
    distances_for_task_pairs: list[OrderedDict[LayerIdentifier, float]] = []
    for b_ in range(B_):
        idx_task1, idx_task2 = indices_task_tuples[b_]
        x1, x2 = X1[idx_task1], X2[idx_task2]
        # x1, x2 = x1.unsqueeze(0), x2.unsqueeze(0)
        # assert x1.size() == torch.Size([1, M, C, H, W]) or x1.size() == torch.Size([1, M, D_])

        # - get distances (index b_) for task pair per layers [1, L]
        print(f'{x1.size()=}, is the input to anatome/dist func [B,M,C,H,W]')
        assert metric_as_sim_or_dist == 'dist'
        dists_task_pair: OrderedDict[LayerIdentifier, float] = dist_data_set_per_layer(f1, f2, x1, x2,
                                                                                       layer_names1, layer_names2,
                                                                                       metric_comparison_type=metric_comparison_type,
                                                                                       iters=iters,
                                                                                       effective_neuron_type=effective_neuron_type,
                                                                                       downsample_method=downsample_method,
                                                                                       downsample_size=downsample_size,
                                                                                       subsample_effective_num_data_method=subsample_effective_num_data_method,
                                                                                       subsample_effective_num_data_param=subsample_effective_num_data_param,
                                                                                       metric_as_sim_or_dist=metric_as_sim_or_dist,
                                                                                       force_cpu=force_cpu
                                                                                       )
        # st()
        # [1, L]
        assert len(dists_task_pair) == len(layer_names1) == len(layer_names2)
        # [b, L] (+) [1, L] -> [b + 1, L]
        distances_for_task_pairs.append(dists_task_pair)  # end result size [B', L]
    # [B', L]
    assert len(distances_for_task_pairs) == B_
    assert len(distances_for_task_pairs[0]) == L
    return distances_for_task_pairs


# legacy, not recommended
def diversity_using_cca_like_metrics(f1: nn.Module, f2: nn.Module,
                                     X1: Tensor, X2: Tensor,
                                     layer_names1: list[str], layer_names2: list[str],
                                     metric_comparison_type: str = 'pwcca',
                                     iters: int = 1,
                                     effective_neuron_type: str = 'filter',
                                     downsample_method: Optional[str] = None,
                                     downsample_size: Optional[int] = None,
                                     subsample_effective_num_data_method: Optional[str] = None,
                                     subsample_effective_num_data_param: Optional[int] = None,
                                     metric_as_sim_or_dist: str = 'dist',
                                     force_cpu: bool = False,

                                     num_tasks_to_consider: int = 25,
                                     consider_diagonal: bool = False
                                     ) -> tuple[OrderedDict, OrderedDict, list[OrderedDict[str, float]]]:
    # ) -> tuple[OrderedDict, OrderedDict, list[OrderedDict[LayerIdentifier, float]]]:
    """
    Div computes as follows:
        - takes in a set of layers [L]
        - a set of batch of tasks data  [B, n*k, C,H,W] (X1 == X2)
        - uses a [B', n*k, C,H,W] of the cross product where B'=num_tasks_to_consider
        - for each layer compute the div value [L, 1]
    Thus signature is:
        div: [L] x [B, n*k, C,H,W]^2 -> [L] x [B', n*k, C,H,W] -> [L, 1]
    note:
        - IMPORTANT: X1 == X2 size is [B, n*k, C,H,W]
        - layer_names = [L]
        - note: for dist(task1, task2) we get [B'] distances. dist: [L]x[B',n*k, C,H,W] ->

    :return: [L]^2, [B', L]
    """
    from anatome.helper import LayerIdentifier, dist_data_set_per_layer, _dists_per_task_per_layer_to_list, \
        compute_stats_from_distance_per_batch_of_data_sets_per_layer
    assert metric_as_sim_or_dist == 'dist'
    # assert args.args.metric_as_sim_or_dist == 'dist'
    assert len(layer_names1) >= 2, f'For now the final and one before final layer are the way to compute diversity'
    L = len(layer_names1)
    B_ = num_tasks_to_consider

    # - [B', L] compute the distance for each task. So you get B distances (from which to compute div) for each layer [L]
    print(f'{X1.size()=}, is the input to anatome/dist func [B,M,C,H,W]')
    distances_for_task_pairs: list[OrderedDict[LayerIdentifier, float]] = get_all_required_distances_for_pairs_of_tasks(
        f1, f2,
        X1, X2,
        layer_names1, layer_names2,
        metric_comparison_type,
        iters,
        effective_neuron_type,
        downsample_method,
        downsample_size,
        subsample_effective_num_data_method,
        subsample_effective_num_data_param,
        metric_as_sim_or_dist,
        force_cpu,

        num_tasks_to_consider,
        consider_diagonal
    )
    # st()
    # [L] x [B, n*k, C,H,W]^2 -> [L] x [B', n*k, C,H,W] -> [B', L]
    assert len(distances_for_task_pairs) == B_
    assert len(distances_for_task_pairs[0]) == L

    # - compute diversity: [B, L] -> [L, 1]^2 (the 2 due to one for div other ci for div)
    div_mu, div_ci = compute_stats_from_distance_per_batch_of_data_sets_per_layer(distances_for_task_pairs)
    print(f'{div_mu=}')
    print(f'{div_ci=}')
    st()
    assert len(div_mu) == L
    assert len(div_ci) == L
    return div_mu, div_ci, distances_for_task_pairs


# def compute_diversity_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks(distances_for_task_pairs: Tensor,
#                                                                                     dist2sim: bool = False):
#     # if dist2sim:
#     #     distances_for_task_pairs: Tensor = 1.0 - distances_for_task_pairs
#     mu, std = distances_for_task_pairs.mean(), distances_for_task_pairs.std()
#     return mu, std

# legacy, not recommended
def compute_diversity_fixed_probe_net_cca_like_metrics(args, meta_dataloader):
    """
    Compute diversity: sample one batch of tasks and use a random cross product of different tasks to compute diversity.

    div: [L] x [B', n*K, C,H,W]^2 -> [L, 1]
    for a set of layers & batch of tasks -> computes div for each layer.
    """
    args.num_tasks_to_consider = args.batch_size
    print(f'{args.num_tasks_to_consider=}')

    L: int = len(args.layer_names)
    B_: int = args.num_tasks_to_consider

    batch = next(iter(meta_dataloader))
    # [B, n*k, C,H,W]
    spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
    # print(f'M = n*k = {spt_x.size(1)=}')
    print(f"M = n*k = {qry_x.size(1)=} (before doing [B, n*k*H*W, D] for [B, N, D'] ")

    # - Compute diversity: [L] x [B', n*K, C,H,W]^2 -> [L, 1]^2, [B, L] i.e. output is div for each layer name
    # assert spt_x.size(0) == args.num_tasks_to_consider
    from copy import deepcopy
    f1 = args.mdl_for_dv
    f2 = args.mdl_for_dv
    assert f1 is f2
    f2 = deepcopy(args.mdl_for_dv)
    # assert they are different objects
    assert f1 is not f2
    # assert they are the same model (e.g. compute if the norm is equal, actually, just check they are the same obj before doing deepcopy)
    assert args.metric_as_sim_or_dist == 'dist', f'Diversity is the expected variation/distance of task, ' \
                                                 f'so you need dist but got {args.metric_as_sim_or_dist=}'
    div_mu, div_ci, distances_for_task_pairs = diversity_using_cca_like_metrics(
        f1=f1, f2=f2, X1=qry_x, X2=qry_x,
        layer_names1=args.layer_names, layer_names2=args.layer_names,
        num_tasks_to_consider=args.num_tasks_to_consider,
        metric_comparison_type=args.metric_comparison_type,
        metric_as_sim_or_dist=args.metric_as_sim_or_dist
    )
    print(f'{args.experiment_option=}')
    print(f'{div_mu=}')
    print(f'{div_ci=}')
    st()
    assert len(div_mu) == L
    assert len(div_ci) == L
    assert len(distances_for_task_pairs) == B_
    assert len(distances_for_task_pairs[0]) == L
    return div_mu, div_ci, distances_for_task_pairs


# - size aware dif
"""
how to take into account the data set size in the diversity coefficient?

- y-axis freq
- x-axis div
- size_aware_div_ceoff(B, smooth) = int_{x \in div} count(x) * div(x) * freq(x) 
	- use kernel density estimation
	- and integrals
- size_aware_div_ceoff(B, discrete with bin_size = X) = sum_{x \in div(bin_size)} count(x) * div(x) * freq(x)
	- discrete sums are fine

- count(x) = 2, idea for count(x) is that when div(x) == 1 it means the tasks at batch named x are maximally different, therefore there are 2 effective counts every time we compute the task2vec distance between tasks (weighted by div, saying)
- general eq
	- sum_{x \in divs} effective_count_concepts(x) * freq(x)
	- where effective_count_concepts(x) = count(x) * div(x)
- this metric came about because
	- according to CLT the data set size N matters to learn things "perfectly"
	- because in practice it assumes it sees all the distribution (maximally diverse), so we want a measure that takes (effective) N into account 
	
ref: 
  - https://www.evernote.com/shard/s410/sh/78688a3a-06bf-a226-541f-a543d4429b3d/97b8576e4fdce577193d367da250e199
"""


def size_aware_div_coef_kernel_density_estimation_kde():
    """
    - y-axis freq
    - x-axis div
    - size_aware_div_ceoff(B, smooth) = int_{x \in div} count(x) * div(x) * freq(x)
        - use kernel density estimation
        - and integrals
    """

    return


def size_aware_div_coef_discrete_histogram_based(distances_as_flat_array: np.array,
                                                 verbose: bool = False,
                                                 count: int = 2,  # count(task2, task2) = 2
                                                 ) -> tuple:
    """
    - y-axis freq
    - x-axis div
    - size_aware_div_ceoff(B, discrete with bin_size = X) = sum_{x \in div(bin_size)} count(x) * div(x) * freq(x)
	    - discrete sums are fine
    """
    from uutils.plot.histograms_uu import get_histogram
    from uutils.plot.histograms_uu import get_x_axis_y_axis_from_seaborn_histogram
    title: str = 'Distribution of Task2Vec Distances'
    ylabel = 'Frequency'
    xlabel: str = 'Cosine Distance between Task Pairs'
    ax = get_histogram(distances_as_flat_array, xlabel, ylabel, title, linestyle=None, color='b')
    x_data, y_data, num_bars_in_histogram, num_bins = get_x_axis_y_axis_from_seaborn_histogram(ax, verbose=True)
    task2vec_dists_binned = x_data
    frequencies_binned = y_data
    if verbose:
        print(f'{x_data, y_data, num_bars_in_histogram, num_bins=}')
        print(f'{frequencies_binned=} \n {frequencies_binned=} \n {sum(frequencies_binned)=}')
    # - close the dummy plot created by get_histogram
    import matplotlib.pyplot as plt
    plt.close()
    # - compute the size aware diversity coefficient
    # numpy dot product between task2vec_dists and frequencies
    size_aware_div_coef: float = float(np.dot(task2vec_dists_binned, frequencies_binned))
    total_frequency: int = int(sum(frequencies_binned))  # not same as size of data set, due to pair wise comparison
    effective_num_tasks: float = count * size_aware_div_coef
    return effective_num_tasks, size_aware_div_coef, total_frequency, task2vec_dists_binned, frequencies_binned, num_bars_in_histogram, num_bins


# - task2vec diversity

def get_task2vec_diversity_coefficient_from_embeddings(embeddings: list[task2vec.Embedding]) -> tuple[float, float]:
    """
    Algorithm sketch:
        - given a list of B tasks e.g. as batches or (spt,qrt) list
        - embed each task using task2vec ~ using fisher information matrix by fine tuning a fixed prob net on the task
        - compute the pairwise cosine distance between each task embedding
        - div, ci = compute the diversity coefficient for the benchmark & confidence interval (ci)
        - return div, ci

    # - get your data
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

    # - compute div
    div_tot = float(distances_as_flat_array.sum())
    print(f'Diversity: {div_tot=}')
    div, ci = get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks(distance_matrix)
    print(f'Diversity: {(div, ci)=}')
    standardized_div: float = get_standardized_diversity_coffecient_from_pair_wise_comparison_of_tasks(distance_matrix)
    print(f'Standardised Diversity: {standardized_div=}')

    import torch
    import random

    class RandomDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __getitem__(self, index):
            data = torch.randn(1, 1) # generate random data
            label = random.randint(0, 1) # generate random label
            return data, label

        def __len__(self):
            return self.num_samples

    dataset = RandomDataset(100) # 100 samples
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    """
    distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
    div, ci = task_similarity.stats_of_distance_matrix(distance_matrix)
    print(f'Diversity: {(div, ci)=}')
    return div, ci


def get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks(distance_matrix: np.ndarray,
                                                                          ) -> tuple[float, float]:
    """
    See docs for: get_task2vec_diversity_coefficient_from_embeddings (avoid copy paste of comments, only maintain 1 comment)
    """
    div, ci = task_similarity.stats_of_distance_matrix(distance_matrix)
    print(f'Diversity: {(div, ci)=}')
    return div, ci


def get_standardized_diversity_coffecient_from_embeddings(embeddings: list[task2vec.Embedding],
                                                          ddof: int = 1,
                                                          ) -> tuple[float, float]:
    """
    Compute the standardized diversity coefficient from a list of task embeddings to ease the comparison of divs
    across benchmarks.

    standardized_mean_metric: float = mean_metric(list_metrics) / unbiased_std_metric(list_metrics)

    ref:
        - https://stats.stackexchange.com/questions/604296/how-does-one-create-comparable-metrics-when-the-original-metrics-are-not-compara?noredirect=1#comment1121965_604296
    """
    distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
    div, unbiased_std = task_similarity.stats_of_distance_matrix(distance_matrix, variance_type='std', ddof=ddof)
    standardized_div: float = div / unbiased_std
    return standardized_div


def get_standardized_diversity_coffecient_from_pair_wise_comparison_of_tasks(distance_matrix: np.ndarray,
                                                                             ddof: int = 1,
                                                                             ) -> float:
    """
    Compute the standardized diversity coefficient from a list of task embeddings to ease the comparison of divs
    across benchmarks.

    standardized_mean_metric: float = mean_metric(list_metrics) / unbiased_std_metric(list_metrics)

    ref:
        - https://stats.stackexchange.com/questions/604296/how-does-one-create-comparable-metrics-when-the-original-metrics-are-not-compara?noredirect=1#comment1121965_604296
    """
    div, unbiased_std = task_similarity.stats_of_distance_matrix(distance_matrix, variance_type='std', ddof=ddof)
    standardized_div: float = div / unbiased_std
    return standardized_div


# - tests

def compute_div_example1_test():
    """
    - sample one batch of tasks and use a random cross product of different tasks to compute diversity.
    """
    mdl: nn.Module = get_default_learner()
    # layer_names: list[str] = get_feature_extractor_conv_layers(include_cls=True)
    layer_names: list[str] = get_last_two_layers(layer_type='conv', include_cls=True)
    args: Namespace = get_minimum_args_for_torchmeta_mini_imagenet_dataloader()
    dataloaders: dict = get_miniimagenet_dataloaders_torchmeta(args)
    for batch_idx, batch_tasks in enumerate(dataloaders['train']):
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch_tasks)
        # - compute diversity
        div_final, div_feature_extractor, div_final_std, div_feature_extractor_std, distances_for_task_pairs = diversity(
            f1=mdl, f2=mdl, X1=qry_x, X2=qry_x, layer_names1=layer_names, layer_names2=layer_names,
            num_tasks_to_consider=2)
        pprint(distances_for_task_pairs)
        print(f'{div_final, div_feature_extractor, div_final_std, div_feature_extractor_std=}')
        break


def compute_div_example2_test():
    # - wrap data laoder in iterator
    # - call next twice to get X1, X2
    # - run div but inclduing diagonal is fine now
    pass


def compute_size_aware_div_test_go():
    # - creating dummy histogram to make sure it doesn't get deleted when we do what to print it
    from uutils.plot.histograms_uu import get_histogram
    data = np.random.normal(size=50)
    ax = get_histogram(data, 'x', 'y', 'title', stat='frequency', linestyle=None, color='r')
    results = size_aware_div_coef_discrete_histogram_based(distances_as_flat_array=np.random.rand(1000), verbose=True)
    print(f'{results=}')
    distances_as_flat_array = np.random.rand(1000)
    effective_num_tasks, size_aware_div_coef, total_frequency, task2vec_dists_binned, frequencies_binned, num_bars_in_histogram, num_bins = size_aware_div_coef_discrete_histogram_based(
        distances_as_flat_array, verbose=True)
    print(
        f'{(effective_num_tasks, size_aware_div_coef, total_frequency, task2vec_dists_binned, frequencies_binned, num_bars_in_histogram, num_bins)=}')
    # - plot the histogram
    import matplotlib.pyplot as plt
    plt.show()
    # - end
    print()


def test_tutorial():
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

    args = Namespace(batch_size=32)
    args.classifier_opts = None
    dataset = RandomDataset(100)  # 100 samples
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    args.dataloaders = {'train': dataloader, 'val': dataloader, 'test': dataloader}
    args.probe_network = nn.Linear(1, 1)
    split = 'train'

    from uutils.torch_uu.metrics.diversity.task2vec_based_metrics.diversity_task2vec.diversity_for_few_shot_learning_benchmark import \
        get_task_embeddings_from_few_shot_dataloader
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
    from uutils.numpy_uu.common import get_diagonal
    distances_as_flat_array, _, _ = get_diagonal(distance_matrix, check_if_symmetric=True)

    # - compute div
    from uutils.torch_uu.metrics.diversity.diversity import \
        get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks, \
        get_standardized_diversity_coffecient_from_pair_wise_comparison_of_tasks

    div_tot = float(distances_as_flat_array.sum())
    print(f'Diversity: {div_tot=}')
    div, ci = get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks(distance_matrix)
    print(f'Diversity: {(div, ci)=}')
    standardized_div: float = get_standardized_diversity_coffecient_from_pair_wise_comparison_of_tasks(distance_matrix)
    print(f'Standardised Diversity: {standardized_div=}')


if __name__ == '__main__':
    import time

    start = time.time()
    # - run
    # compute_div_example1_test()
    # compute_size_aware_div_test_go()
    test_tutorial()
    # - Done
    from uutils import report_times

    print(f"\nSuccess Done!: {report_times(start)}\a")
