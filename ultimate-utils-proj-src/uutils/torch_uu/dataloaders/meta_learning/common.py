# note: decided against this and use the torchmeta loader for data analysis.
# from argparse import Namespace
#
#
# def meta_learning_loader(args: Namespace):
#     """ Get the loader of data for meta-learning, usually, an l2l taskloader,
#     torchmeta dataloader, or even an rfs data loader."""
#     loader = None
#     if 'rfs' in args.data_option or 'torchmeta' in args.data_option:
#         from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloader
#         args.dataloaders: dict = get_meta_learning_dataloader(args)
#         assert hasattr(args, args.dataloaders)
#         assert not hasattr(args, args.tasksets)
#         loader = args.dataloaders
#     elif 'l2l' in args.data_option:
#         from learn2learn.vision.benchmarks import BenchmarkTasksets
#         from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
#         args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)
#         assert not hasattr(args, args.dataloaders)
#         assert hasattr(args, args.tasksets)
#         loader = args.tasksets
#     elif 'mds' in args.data_option:
#         raise NotImplementedError
#     else:
#         raise ValueError(f'Invalid data option: {args.data_option=}')
#     assert loader is not None
#     return loader
