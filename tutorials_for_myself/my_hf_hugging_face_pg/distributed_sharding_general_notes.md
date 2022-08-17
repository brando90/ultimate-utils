#

## Concepts (directly from hf)

src: https://huggingface.co/docs/transformers/perf_train_gpu_many#concepts

Concepts
The following is the brief description of the main concepts that will be described later in depth in this document.

DataParallel (DP) - the same setup is replicated multiple times, and each being fed a slice of the data. The processing is done in parallel and all setups are synchronized at the end of each training step.
TensorParallel (TP) - each tensor is split up into multiple chunks, so instead of having the whole tensor reside on a single gpu, each shard of the tensor resides on its designated gpu. During processing each shard gets processed separately and in parallel on different GPUs and the results are synced at the end of the step. This is what one may call horizontal parallelism, as the splitting happens on horizontal level.
PipelineParallel (PP) - the model is split up vertically (layer-level) across multiple GPUs, so that only one or several layers of the model are places on a single gpu. Each gpu processes in parallel different stages of the pipeline and working on a small chunk of the batch.
Zero Redundancy Optimizer (ZeRO) - Also performs sharding of the tensors somewhat similar to TP, except the whole tensor gets reconstructed in time for a forward or backward computation, therefore the model doesn’t need to be modified. It also supports various offloading techniques to compensate for limited GPU memory.
Sharded DDP - is another name for the foundational ZeRO concept as used by various other implementations of ZeRO.


### Qs

- deepspeed, fsdp, accelerate, ddp, dp, zero,
- ray, pyarrows
- kubernetes, slurm