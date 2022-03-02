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

def fix_args_from_mine_to_rfs(args: Namespace) -> Namespace:
    """
    args: Namespace = Namespace(num_workers=0)
    args.data_root = '~/data/miniImageNet_rfs/miniImageNet'
    args.data_root = Path(args.data_root).expanduser()
    args.data_aug = True
    args.n_ways = 5
    args.n_shots = 5
    args.n_queries = 15
    # args.classes = list(args.data.keys())
    args.n_test_runs = 600  # <- this value might be the reason their CI are smaller than mine, though root grow is slow
    args.n_aug_support_samples = 1
    # args.n_aug_support_samples = 5
    """
    from pathlib import Path
    # args.data_root = '~/data/miniImageNet_rfs/miniImageNet'
    args.data_root = Path(args.data_path).expanduser()
    args.data_aug = True
    args.n_ways = args.n_cls
    args.n_shots = args.k_shots
    args.n_queries = args.k_eval
    # args.classes = list(args.data.keys())
    args.n_test_runs = args.batch_size_eval
    if not hasattr(args, 'n_aug_support_samples'):
        # args.n_aug_support_samples = 1
        # args.n_aug_support_samples = 5
        raise ValueError('You need to provide n_aug_support_samples')
    return args


def get_rfs_meta_learning_mi_dataloader(args: Namespace, ) -> dict:
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
    args.test_batch_size = 1  # default from rfs, DO NOT CHANGE, this is not the number tasks
    args: Namespace = fix_args_from_mine_to_rfs(args)

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
    args.n_test_runs = 600  # <- this value might be the reason their CI are smaller than mine, though root grow is slow
    args.n_aug_support_samples = 1
    # args.n_aug_support_samples = 5

    # - get rfs meta-loaders
    metaloaders: dict = get_rfs_meta_learning_mi_dataloader(args)
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

            # if use_logit:
            #     support_features = net(support_xs).view(support_xs.size(0), -1)
            #     query_features = net(query_xs).view(query_xs.size(0), -1)
            # else:
            #     feat_support, _ = net(support_xs, is_feat=True)
            #     support_features = feat_support[-1].view(support_xs.size(0), -1)
            #     feat_query, _ = net(query_xs, is_feat=True)
            #     query_features = feat_query[-1].view(query_xs.size(0), -1)
            #
            # if is_norm:
            #     support_features = normalize(support_features)
            #     query_features = normalize(query_features)
            #
            # support_features = support_features.detach().cpu().numpy()
            # query_features = query_features.detach().cpu().numpy()
            #
            # support_ys = support_ys.view(-1).numpy()
            # query_ys = query_ys.view(-1).numpy()
            #
            # #  clf = SVC(gamma='auto', C=0.1)
            # if classifier == 'LR':
            #     clf = LogisticRegression(penalty='l2',
            #                              random_state=0,
            #                              C=1.0,
            #                              solver='lbfgs',
            #                              max_iter=1000,
            #                              multi_class='multinomial')
            #     clf.fit(support_features, support_ys)
            #     query_ys_pred = clf.predict(query_features)
            # elif classifier == 'SVM':
            #     clf = make_pipeline(StandardScaler(), SVC(gamma='auto',
            #                                               C=1,
            #                                               kernel='linear',
            #                                               decision_function_shape='ovr'))
            #     clf.fit(support_features, support_ys)
            #     query_ys_pred = clf.predict(query_features)
            # elif classifier == 'NN':
            #     query_ys_pred = NN(support_features, support_ys, query_features)
            # elif classifier == 'Cosine':
            #     query_ys_pred = Cosine(support_features, support_ys, query_features)
            # elif classifier == 'Proto':
            #     query_ys_pred = Proto(support_features, support_ys, query_features, opt)
            # else:
            #     raise NotImplementedError('classifier not supported: {}'.format(classifier))
            #
            # acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
    # return mean_confidence_interval(acc)


if __name__ == '__main__':
    _meta_eval_test()
    print('Done!\a\n')
