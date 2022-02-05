"""
How data sets work in l2l as far as I understand:

you have an object of type BenchmarkTasksets e.g.
    tasksets: BenchmarkTasksets
and you can sample a task from it. A task defined usually a distribution and the objective you want to learn from it.
But in empirical case it means a dataset. So a task is a data set:
    task_data_set_x, task_data_set_y = tasksets.train.sample()
    e.g. [25, 3, 84, 84]
note, 25=k_shots+k_eval.
note, unlike torchmeta, you only sample 1 data set (task) at a time. e.g. via tasksets.<SPLIT>.sample().

Then to get the support & query sets the call the splitter:
        (support_data, support_labels), (query_data, query_labels) = learn2learn.data.partition_task(
            data=task_data_set_x,
            labels=task_data_set_y,
            shots=k_shots,
        )
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
    if args.data_option == 'cifarfs':
        args.tasksets: BenchmarkTasksets = learn2learn.vision.benchmarks.get_tasksets(
            args.data_option,
            train_samples=args.k_shots + args.k_eval,
            train_ways=args.n_classes,
            test_samples=args.k_shots + args.k_eval,
            test_ways=args.n_classes,
            root=args.data_path,
        )
        assert False
    elif args.data_option == 'cifarfs_rfs' or args.data_option == 'fc100_rfs':
        # args.tasksets: BenchmarkTasksets = learn2learn.vision.benchmarks.get_tasksets(
        from uutils.torch_uu.dataloaders.cifar100fs_fc100 import get_tasksets
        args.tasksets: BenchmarkTasksets = get_tasksets(
            args.data_option.split('_')[0],  # returns cifarfs or fc100 string
            train_samples=args.k_shots + args.k_eval,
            train_ways=args.n_classes,
            test_samples=args.k_shots + args.k_eval,
            test_ways=args.n_classes,
            root=args.data_path,
            data_augmentation=args.data_augmentation,
        )
    else:
        raise ValueError(f'Invalid data option, got: {args.data_option}')
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
            task_data_set_x, task_data_set_y = tasksets.train.sample()
            assert task_data_set_x.size() == torch.Size([(args.k_shots + args.k_eval) * args.n_classes, 3, 32, 32])
            assert task_data_set_y.size() == torch.Size([(args.k_shots + args.k_eval) * args.n_classes])
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
            task_data_set_x, task_data_set_y = tasksets.validation.sample()
            assert task_data_set_x.size() == torch.Size([(args.k_shots + args.k_eval) * args.n_classes, 3, 32, 32])
            assert task_data_set_y.size() == torch.Size([(args.k_shots + args.k_eval) * args.n_classes])
            # evaluation_error, evaluation_accuracy = fast_adapt(
            #     batch,
            #     learner,
            #     loss,
            #     adaptation_steps,
            #     shots,
            #     ways,
            #     device,
            # )

            task_data_set_x, task_data_set_y = tasksets.test.sample()
            assert task_data_set_x.size() == torch.Size([(args.k_shots + args.k_eval) * args.n_classes, 3, 32, 32])
            assert task_data_set_y.size() == torch.Size([(args.k_shots + args.k_eval) * args.n_classes])


if __name__ == '__main__':
    l2l_example()
    print('Done!\a')
