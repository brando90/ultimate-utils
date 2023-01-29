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

# - returns list of all task complexities.
def get_task_complexities(embeddings, p=1):
    all_complexities = []

    for embedding in embeddings:
        embedding_hessian = np.array(embedding.hessian)
        embedding_norm = norm(embedding_hessian, ord=1)

        all_complexities += [embedding_norm]

    return all_complexities

# - returns average  and ci of task complexities
def avg_norm_complexity(all_complexities):
    from uutils.torch_uu.metrics.confidence_intervals import mean_confidence_interval
    mu, ci = mean_confidence_interval(all_complexities, confidence=0.95)
    return (mu, ci)

# - returns total complexity of all tasks in benchmark.
def total_norm_complexity(all_complexities):
    return np.sum(all_complexities)


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