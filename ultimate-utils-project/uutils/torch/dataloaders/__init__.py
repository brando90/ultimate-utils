import time
from pathlib import Path

from meta_learning.datasets.mini_imagenet import MetaImageNet, ImageNet

import torch
from torch import nn, nn as nn
from torch.utils.data import DataLoader
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision import transforms as transforms


def process_batch_sl(args, batch):
    batch_x, batch_y = batch
    if torch.cuda.is_available():
        if args.device:
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
        else:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
    return batch_x, batch_y

def get_rfs_sl_dataloader(args):
    args.num_workers = 2 if args.num_workers is None else args.num_workers
    args.target_type = 'classification'
    args.trainin_with_epochs = False
    args.data_root = Path('~/data/miniImageNet_rfs/miniImageNet/').expanduser()
    args.data_aug = True
    args.criterion = nn.CrossEntropyLoss()
    # -- get SL dataloaders
    # train_trans, test_trans = transforms_options[opt.transform]
    # train_sl_loader = DataLoader(ImageNet(args=args, partition=args.split),
    #                              batch_size=args.batch_size, shuffle=True, drop_last=True,
    #                              num_workers=args.num_workers)
    train_sl_loader = DataLoader(ImageNet(args=args, partition='train'),
                                 batch_size=args.batch_size, shuffle=True, drop_last=True,
                                 num_workers=args.num_workers)
    val_sl_loader = DataLoader(ImageNet(args=args, partition='val'),
                               batch_size=args.batch_size // 2, shuffle=False, drop_last=False,
                               num_workers=args.num_workers // 2)
    # -- get meta-dataloaders
    args.n_aug_support_samples = 5  # default from rfs
    # meta_testloader = DataLoader(MetaImageNet(args=args, partition='test'),
    #                              batch_size=args.test_batch_size, shuffle=False, drop_last=False,
    #                              num_workers=args.num_workers)
    meta_valloader = DataLoader(MetaImageNet(args=args, partition='val'),
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers)
    # if opt.use_trainval:
    #     n_cls = 80
    # else:
    #     n_cls = 64
    return train_sl_loader, val_sl_loader, meta_valloader


def get_miniimagenet_dataloaders_torchmeta(args):
    args.trainin_with_epochs = False
    args.data_path = Path('~/data/').expanduser()  # for some datasets this is enough
    args.criterion = nn.CrossEntropyLoss()
    # args.image_size = 84  # do we need this?
    from torchmeta.datasets.helpers import miniimagenet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_augmentation_transforms = transforms.Compose([
        transforms.RandomResizedCrop(84),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize])
    dataset_train = miniimagenet(args.data_path,
                                 transform=data_augmentation_transforms,
                                 ways=args.n_classes, shots=args.k_shots, test_shots=args.k_eval,
                                 meta_split='train', download=True)
    dataset_val = miniimagenet(args.data_path, ways=args.n_classes, shots=args.k_shots, test_shots=args.k_eval,
                               meta_split='val', download=True)
    dataset_test = miniimagenet(args.data_path, ways=args.n_classes, shots=args.k_shots, test_shots=args.k_eval,
                                meta_split='test', download=True)

    meta_train_dataloader = BatchMetaDataLoader(dataset_train, batch_size=args.meta_batch_size_train,
                                                num_workers=args.num_workers)
    meta_val_dataloader = BatchMetaDataLoader(dataset_val, batch_size=args.meta_batch_size_eval,
                                              num_workers=args.num_workers)
    meta_test_dataloader = BatchMetaDataLoader(dataset_test, batch_size=args.meta_batch_size_eval,
                                               num_workers=args.num_workers)
    return meta_train_dataloader, meta_val_dataloader, meta_test_dataloader

def get_transforms_mini_imagenet(args):
    # get transforms for images
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize])
    val_transform = transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize])
    test_transform = transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize])

    return train_transform, val_transform, test_transform

# ---- teats ----

def get_args_for_mini_imagenet():
    from types import SimpleNamespace

    args = SimpleNamespace()
    ## Config for
    args.mode = "meta-train"
    #args.mode = "meta-test"
    args.k_shot = 5
    args.k_eval = 15
    args.n_classes = 5
    args.grad_clip = None # does no gradient clipping if None
    args.grad_clip_mode = None # more specific setting of the crad clipping mode
    ## Episodes params
    args.episodes = 60000
    args.episodes_val = 100
    args.episodes_test = 100
    #args.log_train_freq = 100 if not args.debug else 1
    #args.log_val_freq = 10 if not args.debug else 1
    # careful to have these larger than the size of the meta-set
    args.meta_batch_size_train = 25
    args.meta_batch_size_eval = 4
    ## Inner loop adaptation params
    args.nb_inner_train_steps = 10
    args.track_higher_grads = True # set to false only during meta-testing, but code sets it automatically only for meta-test
    args.copy_initial_weights = False # set to false only if you do not want to train base model's initialization
    ## MAML
    # args.fo = False
    # args.inner_lr = 1e-1
    # args.meta_learner = 'maml_fixed_inner_lr'
    ## Learner/base model options
    args.bn_momentum = 0.95
    args.bn_eps = 1e-3
    args.base_model_mode = 'child_mdl_from_opt_as_a_mdl_for_few_shot_learning_paper'
    ## miniImagenet options
    args.data_root = Path("~/automl-meta-learning/data/miniImagenet").expanduser()
    args.n_workers = 4
    args.pin_memory = False # it is generally not recommended to return CUDA tensors in multi-process loading because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing (see CUDA in multiprocessing). Instead, we recommend using automatic memory pinning (i.e., setting pin_memory=True), which enables fast data transfer to CUDA-enabled GPUs. https://pytorch.org/docs/stable/data.html
    args.criterion = nn.CrossEntropyLoss()
    args.image_size = 84
    return args


def test_torchmeta_good_accumulator():
    import torch
    import torch.optim as optim
    from automl.child_models.learner_from_opt_as_few_shot_paper import Learner
    import higher

    ## get args for test
    args = get_args_for_mini_imagenet()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## get base model that meta-lstm/maml use
    base_model = Learner(image_size=args.image_size, bn_eps=args.bn_eps, bn_momentum=args.bn_momentum, n_classes=args.n_classes).to(args.device)

    ## get meta-set
    meta_train_loader, _, _ = get_meta_set_loaders_miniImagenet(args)

    ## start episodic training
    meta_params = base_model.parameters()
    outer_opt = optim.Adam(meta_params, lr=1e-2)
    base_model.train()
    for episode, (spt_x, spt_y, qry_x, qry_y) in enumerate(meta_train_loader):
        assert(spt_x.size(1) == args.k_shot*args.n_classes)
        assert(qry_x.size(1) == args.k_eval*args.n_classes)
        ## Get Inner Optimizer (for maml)
        inner_opt = torch.optim.SGD(base_model.parameters(), lr=1e-1)
        ## Accumulate gradient of meta-loss wrt fmodel.param(t=0)
        nb_tasks = spt_x.size(0)
        meta_losses, meta_accs = [], []
        assert(nb_tasks == args.meta_batch_size_train)
        for t in range(nb_tasks):
            ## Get supprt & query set for the current task
            spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]
            ## Inner Loop Adaptation
            with higher.innerloop_ctx(base_model, inner_opt, copy_initial_weights=args.copy_initial_weights, track_higher_grads=args.track_higher_grads) as (fmodel, diffopt):
                for i_inner in range(args.nb_inner_train_steps):
                    fmodel.train()
                    # base/child model forward pass
                    spt_logits_t = fmodel(spt_x_t)
                    inner_loss = args.criterion(spt_logits_t, spt_y_t)
                    # inner-opt update
                    diffopt.step(inner_loss)
                    inner_loss = args.criterion(spt_logits_t, spt_y_t)
            ## Evaluate on query set for current task
            qry_logits_t = fmodel(qry_x_t)
            qry_loss_t = args.criterion(qry_logits_t,  qry_y_t)
            ## Accumulate gradients wrt meta-params for each task
            qry_loss_t.backward() # note this is memory efficient
            ## collect losses & accs for logging/debugging
            meta_losses.append(qry_loss_t.detach()) # remove history so it be memory efficient and able to print stats
        ## do outer step
        outer_opt.step()
        outer_opt.zero_grad()
        print(f'[episode={episode}] meta_loss = {sum(meta_losses)/len(meta_losses)}')

if __name__ == '__main__':
    from uutils import report_times
    start = time.time()
    test_torchmeta_good_accumulator()
    time_passed_msg, _, _, _ = report_times(start)
    print(time_passed_msg)