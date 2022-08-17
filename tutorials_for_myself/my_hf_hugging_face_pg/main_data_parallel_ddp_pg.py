"""

- training on multiple gpus: https://huggingface.co/docs/transformers/perf_train_gpu_many#efficient-training-on-multiple-gpus
- data paralelism, dp vs ddp: https://huggingface.co/docs/transformers/perf_train_gpu_many#data-parallelism
- github example: https://github.com/huggingface/transformers/tree/main/examples/pytorch#distributed-training-and-mixed-precision
    - above came from hf discuss: https://discuss.huggingface.co/t/using-transformers-with-distributeddataparallel-any-examples/10775/7
"""
#%%