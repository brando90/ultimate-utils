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

