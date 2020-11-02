import torch
import torch.optim as optim
import torch.nn as nn

from torchmeta.datasets import MiniImagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets.helpers import miniimagenet

import torchvision.transforms as transforms

from pathlib import Path

from types import SimpleNamespace

def get_transforms_mini_imagenet(args):
    ## get transforms for images
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

def get_meta_set_loaders_miniImagenet(args, download=False):
    download = args.download or download if hasattr(args, 'download') else download
    ## get transforms
    train_transform, val_transform, test_transform = get_transforms_mini_imagenet(args)

    ## get mini-Imagenet data-set for train, val, test
    train_miniImagenet = miniimagenet(
        args.data_root, 
        ways=args.n_classes, 
        shots=args.k_shot, 
        test_shots=args.k_eval, 
        meta_split='train', 
        transform=train_transform, 
        download=download)
    val_miniImagenet = miniimagenet(
        args.data_root, 
        ways=args.n_classes, 
        shots=args.k_shot, 
        test_shots=args.k_eval, 
        meta_split='val', 
        transform=val_transform, 
        download=download)
    test_miniImagenet = miniimagenet(
        args.data_root, 
        ways=args.n_classes, 
        shots=args.k_shot, 
        test_shots=args.k_eval, 
        meta_split='test', 
        transform=test_transform, 
        download=download)

    ## get batch meta dataloaders for meta-train, meta-val, meta-test
    meta_train_loader = BatchMetaDataLoader(
        train_miniImagenet, 
        batch_size=args.meta_batch_size_train, 
        num_workers=args.n_workers)
    meta_val_loader = BatchMetaDataLoader(
        val_miniImagenet, 
        batch_size=args.meta_batch_size_eval, 
        num_workers=args.n_workers)
    meta_test_loader = BatchMetaDataLoader(
        test_miniImagenet, 
        batch_size=args.meta_batch_size_eval, 
        num_workers=args.n_workers)

    return meta_train_loader, meta_val_loader, meta_test_loader

def get_args_for_mini_imagenet():
    args = SimpleNamespace()
    ## Config
    args.mode = "meta-train"
    #args.mode = "meta-test"
    args.k_shot = 1
    args.k_eval = 15
    args.n_classes = 5
    args.grad_clip = None # does no gradient clipping if None
    args.grad_clip_mode = None # more specific setting of the crad clipping mode
    ## Episodes params
    args.episodes = 2
    args.episodes_val = 1
    args.episodes_test = 1
    #args.log_train_freq = 100 if not args.debug else 1
    #args.log_val_freq = 10 if not args.debug else 1
    # careful to have these larger than the size of the meta-set
    args.meta_batch_size_train = 3
    args.meta_batch_size_eval = 2
    ## Inner loop adaptation params
    args.nb_inner_train_steps = 2
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
    args.download = True
    args.data_root = Path("~/automl-meta-learning/data/miniImagenet").expanduser()
    args.n_workers = 4
    args.pin_memory = False # it is generally not recommended to return CUDA tensors in multi-process loading because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing (see CUDA in multiprocessing). Instead, we recommend using automatic memory pinning (i.e., setting pin_memory=True), which enables fast data transfer to CUDA-enabled GPUs. https://pytorch.org/docs/stable/data.html
    args.criterion = nn.CrossEntropyLoss()
    args.image_size = 84
    return args

def test_torchmeta_good_accumulator():
    from tqdm import tqdm

    from automl.child_models.learner_from_opt_as_few_shot_paper import Learner
    import automl.child_models.learner_from_opt_as_few_shot_paper

    import higher

    ## get args for test
    args = get_args_for_mini_imagenet()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## get base model that meta-lstm/maml use
    base_model = Learner(
        image_size=args.image_size, 
        bn_eps=args.bn_eps, 
        bn_momentum=args.bn_momentum, 
        n_classes=args.n_classes).to(args.device)
    
    ## get meta-set 
    meta_train_loader, _, _ = get_meta_set_loaders_miniImagenet(args)

    ## start episodic training
    meta_params = base_model.parameters()
    outer_opt = optim.Adam(meta_params, lr=1e-1)
    base_model.train()

    print('\nTraining starting...')
    with tqdm(meta_train_loader, total=args.episodes) as metaset_dataloader:
        print(f'len(metaset_dataloader) = {len(metaset_dataloader)}')
        print(f'args.episodes = {args.episodes}')
        for episode, batch in enumerate(metaset_dataloader):
            if episode > args.episodes:
                break
            spt_x, spt_y = batch["train"]
            qry_x, qry_y = batch["test"]
            print(f'spt_x.size() = {spt_x.size()}')
            print(f'spt_y.size() = {spt_y.size()}')
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
    import time
    from uutils import report_times

    start = time.time()
    test_torchmeta_good_accumulator()
    time_passed_msg, _, _, _ = report_times(start)
    print(time_passed_msg)