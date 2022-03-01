"""
note: since rfs mainly uses this code to evaluate models -- since they trian with (union) supervised learning
(not episodic meta-learning). That is the entire point of their paper!
"""

from argparse import Namespace

import torch
from torch.utils.data import DataLoader

from uutils.torch_uu.dataloaders.meta_learning.transforms_cfg_rfs import transforms_options

from uutils.torch_uu.dataset.rfs_mini_imagenet import ImageNet, MetaImageNet


# from dataset.mini_imagenet import ImageNet, MetaImageNet

# from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
# from dataset.cifar import CIFAR100, MetaCIFAR100
# from dataset.transform_cfg import transforms_options, transforms_list

def rfs_meta_learning_mi_dataloader(args: Namespace, ) -> dict:
    """
    return a normal pytorch data loader,

    ref: https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/eval_fewshot.py#L88
    """
    # opt = parse_option()

    # test loader
    # args = opt
    # args.batch_size = args.test_batch_size
    # args.n_aug_support_samples = 1

    # if args.dataset == 'miniImageNet':
    args.transform = 'A'  # default for miniImagenet
    args.test_batch_size = 1  # default from rfs

    # - get rfs meta-loaders
    train_trans, test_trans = transforms_options[args.transform]
    # gives error that pickle file doesn't exist
    # meta_trainloader = DataLoader(MetaImageNet(args=args, partition='train',
    #                                            train_transform=train_trans,
    #                                            test_transform=test_trans,
    #                                            fix_seed=False),
    #                               batch_size=args.test_batch_size, shuffle=False, drop_last=False,
    #                               num_workers=args.num_workers)
    meta_trainloader = None
    meta_valloader = DataLoader(MetaImageNet(args=args, partition='val',
                                             train_transform=train_trans,
                                             test_transform=test_trans,
                                             fix_seed=False),
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers)
    meta_testloader = DataLoader(MetaImageNet(args=args, partition='test',
                                              train_transform=train_trans,
                                              test_transform=test_trans,
                                              fix_seed=False),
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.num_workers)
    # these args seem they use it to create model
    # if args.use_trainval:
    #     n_cls = 80
    # else:
    #     n_cls = 64
    # model = create_model(opt.model, n_cls, opt.dataset)
    # ckpt = torch.load(opt.model_path)
    # model.load_state_dict(ckpt['model'])
    #
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     cudnn.benchmark = True
    # --
    # meta_trainloader = meta_valloader
    dataloaders = {'train': meta_trainloader, 'val': meta_valloader, 'test': meta_testloader}
    return dataloaders


# def main():
#     opt = parse_option()
#
#     # test loader
#     args = opt
#     args.batch_size = args.test_batch_size
#     # args.n_aug_support_samples = 1
#
#     if opt.dataset == 'miniImageNet':
#         train_trans, test_trans = transforms_options[opt.transform]
#         meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
#                                                   train_transform=train_trans,
#                                                   test_transform=test_trans,
#                                                   fix_seed=False),
#                                      batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
#                                      num_workers=opt.num_workers)
#         meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
#                                                  train_transform=train_trans,
#                                                  test_transform=test_trans,
#                                                  fix_seed=False),
#                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
#                                     num_workers=opt.num_workers)
#         if opt.use_trainval:
#             n_cls = 80
#         else:
#             n_cls = 64
#     elif opt.dataset == 'tieredImageNet':
#         train_trans, test_trans = transforms_options[opt.transform]
#         meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
#                                                         train_transform=train_trans,
#                                                         test_transform=test_trans,
#                                                         fix_seed=False),
#                                      batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
#                                      num_workers=opt.num_workers)
#         meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
#                                                        train_transform=train_trans,
#                                                        test_transform=test_trans,
#                                                        fix_seed=False),
#                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
#                                     num_workers=opt.num_workers)
#         if opt.use_trainval:
#             n_cls = 448
#         else:
#             n_cls = 351
#     elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
#         train_trans, test_trans = transforms_options['D']
#         meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
#                                                   train_transform=train_trans,
#                                                   test_transform=test_trans,
#                                                   fix_seed=False),
#                                      batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
#                                      num_workers=opt.num_workers)
#         meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
#                                                  train_transform=train_trans,
#                                                  test_transform=test_trans,
#                                                  fix_seed=False),
#                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
#                                     num_workers=opt.num_workers)
#         if opt.use_trainval:
#             n_cls = 80
#         else:
#             if opt.dataset == 'CIFAR-FS':
#                 n_cls = 64
#             elif opt.dataset == 'FC100':
#                 n_cls = 60
#             else:
#                 raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
#     else:
#         raise NotImplementedError(opt.dataset)
#
#     # load model
#     model = create_model(opt.model, n_cls, opt.dataset)
#     ckpt = torch.load(opt.model_path)
#     model.load_state_dict(ckpt['model'])
#
#     if torch.cuda.is_available():
#         model = model.cuda()
#         cudnn.benchmark = True

# - test

def _meta_eval_test():
    from tqdm import tqdm
    from pathlib import Path

    args: Namespace = Namespace(num_workers=0)
    args.data_root = '~/data/miniImageNet_rfs/miniImageNet'
    args.data_root = Path(args.data_root).expanduser()
    args.data_aug = True
    args.n_ways = 5
    args.n_shots = 5
    args.n_queries = 15
    # args.classes = list(args.data.keys())
    args.n_test_runs = 600
    # args.n_aug_support_samples = 1
    args.n_aug_support_samples = 5

    # - get rfs meta-loaders
    metaloaders: dict = rfs_meta_learning_mi_dataloader(args)
    testloader = metaloaders['test']

    # - run their eval code
    acc = []

    with torch.no_grad():
        print(f'{len(testloader)=}')
        for idx, data in tqdm(enumerate(testloader)):
            support_xs, support_ys, query_xs, query_ys = data
            if torch.cuda.is_available():
                support_xs = support_xs.cuda()
                query_xs = query_xs.cuda()
            batch_size, _, channel, height, width = support_xs.size()
            support_xs = support_xs.view(-1, channel, height, width)
            query_xs = query_xs.view(-1, channel, height, width)
            print()
            print(f'{support_xs.size()=}')
            print(f'{query_xs.size()=}')
            print()

            # acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
    # return mean_confidence_interval(acc)


if __name__ == '__main__':
    _meta_eval_test()
    print('Done!\a\n')
