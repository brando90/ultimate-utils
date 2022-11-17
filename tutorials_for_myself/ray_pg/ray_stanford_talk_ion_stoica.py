"""

ref:
    - ray main website getting started guide: https://docs.ray.io/en/latest/ray-overview/index.html

Notes:
Ray is a unified way to scale Python and AI applications from a laptop to a cluster.
With Ray, you can seamlessly scale the same code from a laptop to a cluster. Ray is designed to be general-purpose, meaning that it can performantly run any kind of workload. If your application is written in Python, you can scale it with Ray, no other infrastructure required

Moore's law: roughly every two years, the number of transistors on microchips will double. Commonly referred to as Moore's Law, this phenomenon suggests that computational progress will become significantly faster, smaller, and more efficient over time.

palm 540 B, ~19 years to train in a single chip, seems no way out but distributed training.

Goal of Ray: unified framework for distributed computing (note since each stage of processing is different with different
apis so harf to use stich toegether,
preprocessing, training, tuning, batch prediction, servingd).

ray: passes data by reference vs rpc which does by value and passes data less in the network (e.g. btw nodes).

note: ray is the "backend" for distributed training to cross node boundary. While nccl is for multigpu.

(spark is in java/scala...?!, why? ask pedro)

- even expensive to move data from ram to gpu!?

lessons learned ray (what users want most):

ray basics:
    future:
    - actor: Actors extend the Ray API from functions (tasks) to classes. An actor is essentially a stateful worker (or a service)
    When a new actor is instantiated, a new worker is created, and methods of the actor are scheduled on that specific worker and can access and mutate the state of that worker.
    https://docs.ray.io/en/latest/ray-core/actors.html
    example: counter class from lec
    task: Ray enables arbitrary functions to be executed asynchronously on separate Python workers. Such functions are called Ray remote functions and their asynchronous invocations are called Ray tasks.
    - example: ...

Questions:
Q1: when using ray and you want to train across multiple GPUs in beesides/addition to multiple nodes (which seems
to be rays main use/strength), how does that work? is the multiple GPU handled by nccl or ray itself? e.g.
I distribute my LLM across GPUs but the data is distributed across nodes.
Q2: Jax vs ray, does this work?
Q3: ray's interaction with slurm, kubernetes?
Q4: is only tf expensive to initialize? what about pytorch or jax? (why is this?)
Q5: why do you care about fault tolerance? ML ppl I assume wouldn't care, the methods are supposed to be robust
to noise, this is just yet another source of noise...
Q6: related to previous, I missed the main lesson learned for you which was something about users of ray not
caring about fault tolerance. Do you mind repeating that to me? (sorry you went to fast through that slide).
Also, if your open to sharing the slides that would be awesome!
Q7: why doesn't moore's law apply for memory? at least ram and gpu memory? https://discuss.ray.io/t/why-doesnt-moores-law-apply-for-memory-at-least-ram-and-gpu-memory-how-does-this-affect-ray/8339

"""
#%%

#%%