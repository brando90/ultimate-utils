# Computing metrics & Tutorials

## Computing diversity tutorial (using batch & (support, queary) sets)

```
    # -- get your data!
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
```

## Computing complexity tutorial (using batch & (support, queary) sets)

```
    # - get your data
    embeddings: list[task2vec.Embedding] = get_random_data_todo()

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
