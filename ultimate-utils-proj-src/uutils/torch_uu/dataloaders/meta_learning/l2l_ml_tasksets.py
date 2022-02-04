"""
How data sets work in l2l as far as I understand:

you have an object of type BenchmarkTasksets e.g.
    tasksets
and you can sample a task from it


"""
from argparse import Namespace
from pathlib import Path

import learn2learn
import torch
from learn2learn.vision.benchmarks import BenchmarkTasksets

def get_all_l2l_official_benchmarks_supported() -> list:
    """
    dict_keys(['omniglot', 'mini-imagenet', 'tiered-imagenet', 'fc100', 'cifarfs'])
    """
    import learn2learn as l2l
    benchmark_names: list = list(l2l.vision.benchmarks.list_tasksets())
    return benchmark_names

def get_l2l_tasksets(args: Namespace) -> BenchmarkTasksets:
    args.tasksets: BenchmarkTasksets = learn2learn.vision.benchmarks.get_tasksets(
        args.data_option,
        train_samples=args.k_shots,
        train_ways=args.n_classes,
        test_samples=args.k_eval,
        test_ways=args.n_classes,
        root=args.data_path,
    )
    return args.tasksets

# - tests

def l2l_example(meta_batch_size: int = 4, num_iterations: int = 5):
    args: Namespace = Namespace(k_shots=5, n_classes=5, k_eval=15, data_option='cifarfs',
                                data_path=Path('~/data/l2l_data/'))

    tasksets: BenchmarkTasksets = get_l2l_tasksets(args)
    for iteration in range(num_iterations):
        # opt.zero_grad()
        for task in range(meta_batch_size):
            # Compute meta-training loss
            # learner = maml.clone()
            #
            support_x, support_y = tasksets.train.sample()
            assert support_x.size() == torch.Size([args.k_shots*args.n_classes, 3, 32, 32])
            assert support_y.size() == torch.Size([args.k_shots*args.n_classes])
            # evaluation_error, evaluation_accuracy = fast_adapt(
            #     batch,
            #     learner,
            #     loss,
            #     adaptation_steps,
            #     shots,
            #     ways,
            #     device,
            # )
            # evaluation_error.backward()

            # Compute meta-validation loss
            # learner = maml.clone()
            support_x, support_y = tasksets.validation.sample()
            assert support_x.size() == torch.Size([args.k_eval*args.n_classes, 3, 32, 32])
            assert support_y.size() == torch.Size([args.k_eval*args.n_classes])
            # evaluation_error, evaluation_accuracy = fast_adapt(
            #     batch,
            #     learner,
            #     loss,
            #     adaptation_steps,
            #     shots,
            #     ways,
            #     device,
            # )

            support_x, support_y = tasksets.test.sample()
            assert support_x.size() == torch.Size([args.k_eval*args.n_classes, 3, 32, 32])
            assert support_y.size() == torch.Size([args.k_eval*args.n_classes])


if __name__ == '__main__':
    l2l_example()
    print('Done!\a')