'''
We propose to measure the complexity of a benchmark, by taking the sum or average of the Lp norm of Task2Vec embeddings.
Either:
1. Average complexity: complexity_avg(p) = 1/{number of tasks} sum_{task \in benchmark} ||F_task||_p
2. Total complexity: complexity_total(p) = sum_{task \in benchmark} ||F_task||_p

According to Fig 2 in Task2vec paper https://openaccess.thecvf.com/content_ICCV_2019/papers/Achille_Task2Vec_Task_Embedding_for_Meta-Learning_ICCV_2019_paper.pdf
the L1 norm of a task is strongly correlated to the test error of the task in iNaturalist and CUB.
Hence we set our default norm to be p = 1.

Ideas (according to brando's recording msg):
Also, total diversity of a benchmark
Also, weighted diversity, not very obvious.
Multiply total diversity with total complexity  or average diversity with average complexity

more difficult:
Also some type of cross-benchmark comparison complexity(?)
'''

from numpy.linalg import norm
import numpy as np

from uutils.torch_uu.metrics.diversity.task2vec_based_metrics import task2vec, task_similarity


# - returns list of all task complexities.
def get_task_complexities(embeddings: list[task2vec.Embedding], p: int = 1) -> list[float]:
    all_complexities: list[float] = []
    for embedding in embeddings:
        embedding_hessian = np.array(embedding.hessian)
        embedding_norm = norm(embedding_hessian, ord=p)
        all_complexities += [embedding_norm]
    return all_complexities


# - returns average  and ci of task complexities
def avg_norm_complexity(all_complexities, confidence: float = 0.95):
    from uutils.torch_uu.metrics.confidence_intervals import mean_confidence_interval
    mu, ci = mean_confidence_interval(all_complexities, confidence=confidence)
    return mu, ci


# - returns total complexity of all tasks in benchmark.
def total_norm_complexity(all_complexities):
    return np.sum(all_complexities)


def standardized_norm_complexity(embeddings: list[task2vec.Embedding], p: int = 1) -> float:
    """

    standardized_mean_metric: float = mean_metric(list_metrics) / unbiased_std_metric(list_metrics)

    Notes:
        - In general, the degrees of freedom of an estimate of a parameter are equal to the number of independent scores
         that go into the estimate minus the number of parameters used as intermediate steps in the estimation of the
         parameter itself.
           - num_independent_scores (input num vars) - num_vars_used_to_compute_statistic (num vars used) = degrees of freedom
        - For example, if the variance is to be estimated from a random sample of N independent scores, then the degrees
         of freedom is equal to the number of independent scores (N) minus the number of parameters estimated as
         intermediate steps (one, namely, the sample mean) and is therefore equal to N − 1.
    """
    all_complexities: list[float] = get_task_complexities(embeddings, p)
    avg_complexity, ci = avg_norm_complexity(all_complexities)
    unbiased_std_complexity: float = float(np.std(all_complexities, ddof=1))
    standardized_complexity: float = avg_complexity / unbiased_std_complexity
    return float(standardized_complexity)


'''
def avg_norm_complexity(embeddings, p=1):
    #print("embeddings: ", embeddings)
    avg_complexity = 0
    all_complexities = []

    for embedding in embeddings:
        embedding_hessian = np.array(embedding.hessian)
        embedding_norm = norm(embedding_hessian, ord = 1)

        avg_complexity += embedding_norm
        all_complexities += [embedding_norm]
    avg_complexity /= len(embeddings)

    return avg_complexity, all_complexities



def total_norm_complexity(embeddings, p=1):
    avg_complexity, all_complexities = avg_norm_complexity(embeddings, p)
    total_complexity = avg_complexity * len(embeddings)

    return total_complexity, all_complexities
'''
