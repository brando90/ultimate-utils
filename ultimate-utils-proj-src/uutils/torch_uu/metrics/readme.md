# Computing metrics & Tutorials

## Computing diversity tutorial (using batch & (support, queary) sets)

```
    # - get your data
    embeddings: list[task2vec.Embedding] = get_random_data_todo()

    # - compute distance matrix & task2vec based diversity, to demo` task2vec, this code computes pair-wise distance between task embeddings
    distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
    print(f'{distance_matrix=}')
    distances_as_flat_array, _, _ = get_diagonal(distance_matrix, check_if_symmetric=True)
    
    # - compute div
        from uutils.torch_uu.metrics.diversity.diversity import get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks, get_standardized_diversity_coffecient_from_pair_wise_comparison_of_tasks
    div_tot = float(distances_as_flat_array.sum())
    print(f'Diversity: {div_tot=}')
    div, ci = get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks(distance_matrix)
    print(f'Diversity: {(div, ci)=}')
    standardized_div: float = get_standardized_diversity_coffecient_from_pair_wise_comparison_of_tasks(distance_matrix)
    print(f'Standardised Diversity: {standardized_div=}')
```
todo: make it run on random data. 

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
    standardized_norm_complexity = standardized_norm_complexity(all_complexities)
    print(f'Standardized Norm Complexity: {standardized_norm_complexity=}')
```
