"""


"""

from __future__ import division, print_function, absolute_import

import os
import re
import pdb
import glob
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import PIL.Image as PILI
import numpy as np
import random

from pathlib import Path

from types import SimpleNamespace

from tqdm import tqdm

class MetaSet(data.Dataset):

    def __init__(self, root, phase='train', k_shot=5, k_eval=15, transform=None):
        ## Locate meta-set & tasl labels/idx i.e. get root to data-sets D_t for each task
        root = os.path.join(root, phase) # e.g '/Users/brandomiranda/automl-meta-learning/data/miniImagenet/train'
        # count the number of classes
        self.labels = sorted(os.listdir(root)) # e.g. 64 for meta-train, 16 for meta-val, 20 for meta-test
        
        ## Get meta-set
        meta_set_as_paths = [] # e.g. holds all the paths to the images e.g. 64 tasks/classes of mini-imagenet
        for label in self.labels: # for each label get the 600 paths to the images
            ## Append path to Dt to meta_set
            # get e. path to D_t with 600
            path_to_Dt = os.path.join(root, label, '*') # e.g. '/Users/brandomiranda/automl-meta-learning/data/miniImagenet/train/n13133613/*' note * is to match all names of images
            # get path to all (600) images to current images of D_t. Note: glog.glob = Return a possibly-empty list of path names that match pathname, since at the end we have * we match all images.
            list_pathnames_of_images = glob.glob( path_to_Dt ) # e.g. ['/Users/brandomirand...00181.jpg', '/Users/brandomirand...00618.jpg', '/Users/brandomirand...00624.jpg', '/Users/brandomirand...00630.jpg', '/Users/brandomirand...00397.jpg', '/Users/brandomirand...01089.jpg', '/Users/brandomirand...00368.jpg', '/Users/brandomirand...00340.jpg', '/Users/brandomirand...00432.jpg', '/Users/brandomirand...00591.jpg', '/Users/brandomirand...01102.jpg', '/Users/brandomirand...00234.jpg', '/Users/brandomirand...00220.jpg', '/Users/brandomirand...00546.jpg', ...]
            meta_set_as_paths.append( list_pathnames_of_images ) # append list of 600 paths to images for Dt
        # meta_set_as_paths = [Path_Dt]^64_t=1 where |D_t| = 600

        ## Generate list of dataloaders for each Class-data set of size 600. With the dataloader for each class Dataset we can sample a N-way K-shot task of size N*K
        self.meta_set = [] # {D_t}^64_t data set of data sets
        #self.meta_set_loader = [] # {DL_t}^64_t set of for sampling N*K instances from D_t
        for idx, _ in enumerate(self.labels):
            ## wrap each D_t in a class Dataset
            list_pathnames_of_images = meta_set_as_paths[idx] # list for 600 images for the current task
            Dt = ClassDataset(images=list_pathnames_of_images, label=idx, transform=transform) # D_t
            self.meta_set.append(Dt)
            ## wrap D_t in a Dataloader for sampling N*K instances
            #DLt = data.DataLoader(Dt, batch_size= k_shot+k_eval, shuffle=True, num_workers=0) # data loader for the current task D_t for class/label t
            #self.meta_set_loader.append(DL) # collect all DL_t e.g. sizes 64, 16, 20
        print(f'len(self.meta_set) = {len(self.meta_set)}')

    def __getitem__(self, idx):
        Dt = self.meta_set[idx]
        return Dt

    def __len__(self):
        """ Returns the number of data set in D_t.

        Returns:
            [int]: number of tasks e.g. 64 for mini-Imagenet's meta-train-set
        """
        return len(self.labels) # e.g. 64 for mini-Imagenet's meta-train-set 

class ClassDataset(data.Dataset):

    def __init__(self, images, label, transform=None):
        """Args:
            images (list of str): each item is a path to an image of the same label
            label (int): the label of all the images
        """
        self.images = images # e.g. ['/Users/brandomirand...00106.jpg', '/Users/brandomirand...01218.jpg', '/Users/brandomirand...00660.jpg', '/Users/brandomirand...00674.jpg', '/Users/brandomirand...00884.jpg', '/Users/brandomirand...00890.jpg', '/Users/brandomirand...00648.jpg', '/Users/brandomirand...01230.jpg', '/Users/brandomirand...00338.jpg', '/Users/brandomirand...00489.jpg', '/Users/brandomirand...00270.jpg', '/Users/brandomirand...00264.jpg', '/Users/brandomirand...01152.jpg', '/Users/brandomirand...00258.jpg', ...]
        self.label = label # e.g. 37
        self.transform = transform

    def __getitem__(self, idx):
        """
        Args:
            idx ([int]): an index from D_t e.g. 261

        Returns:
            [tensor, int]: single example (x,y) where y is an int.
        """
        ## get a single image (and transform it if necessary)
        image = PILI.open(self.images[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.label 

    def __len__(self):
        """ 
        Returns:
            [int]: number of images x in D_t
        """
        return len(self.images) # e.g. 600

class EpisodicSampler(data.Sampler):
    """
    Comments:
        Meant to be passed to batch_sampler when creating a data_loader.
        The way it's implemented is meant simply to keep track of the episode we are on and halt the meta-set loader when
        we created all the episodes we wanted.
    """

    def __init__(self, total_classes, n_episodes):
        self.total_classes = total_classes # e.g. 64, 16, 20 for mini-imagenet (total number of tasks for a meta-set)
        self.n_episodes = n_episodes # number of times to samples from a task D_t e,g, 60K for MAML/meta-lstm.

    def __iter__(self):
        for episode in range(self.n_episodes):
            # return all classes so we can create M batches of N-way,K-shot tasks in the collate function
            # the yield acts like a state machine keeping track of which episode we are on
            yield range(self.n_episodes)

    def __len__(self):
        """Returns the number of episodes.

        Returns:
            [int]: returns the number of times we will sample episodes
        """
        return self.n_episode

class GetMetaBatch_NK_WayClassTask:

    def __init__(self, meta_batch_size, n_classes, k_shot, k_eval, shuffle=True, pin_memory=True, original=False, flatten=True):
        self.meta_batch_size = meta_batch_size
        self.n_classes = n_classes
        self.k_shot = k_shot
        self.k_eval = k_eval
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.original = original
        self.flatten = flatten

    def __call__(self, all_datasets, verbose=False):
        NUM_WORKERS = 0 # no need to change
        get_data_loader = lambda data_set: iter(data.DataLoader(data_set, batch_size=self.k_shot+self.k_eval, shuffle=self.shuffle, num_workers=NUM_WORKERS, pin_memory=self.pin_memory))
        #assert( len(meta_set) == self.meta_batch_size*self.n_classes )
        # generate M N,K-way classification tasks
        batch_spt_x, batch_spt_y, batch_qry_x, batch_qry_y = [], [], [], []
        for m in range(self.meta_batch_size):
            n_indices = random.sample(range(0,len(all_datasets)), self.n_classes)
            # create N-way, K-shot task instance
            spt_x, spt_y, qry_x, qry_y = [], [], [], []
            for i,n in enumerate(n_indices):
                data_set_n = all_datasets[n]
                dataset_loader_n = get_data_loader(data_set_n) # get data set for class n
                data_x_n, data_y_n = next(dataset_loader_n) # get all data from current class 
                spt_x_n, qry_x_n = data_x_n[:self.k_shot], data_x_n[self.k_shot:] # [K, CHW], [K_eval, CHW]
                # get labels
                if self.original:
                    #spt_y_n = torch.tensor([n]).repeat(self.k_shot)
                    #qry_y_n = torch.tensor([n]).repeat(self.k_eval)
                    spt_y_n, qry_y_n = data_y_n[:self.k_shot], data_y_n[self.k_shot:]
                else:
                    spt_y_n = torch.tensor([i]).repeat(self.k_shot)
                    qry_y_n = torch.tensor([i]).repeat(self.k_eval)
                # form K-shot task for current label n
                spt_x.append(spt_x_n); spt_y.append(spt_y_n) # array length N with tensors size [K, CHW]
                qry_x.append(qry_x_n); qry_y.append(qry_y_n) # array length N with tensors size [K, CHW]
            # form N-way, K-shot task with tensor size [N,W, CHW]
            spt_x, spt_y, qry_x, qry_y = torch.stack(spt_x), torch.stack(spt_y), torch.stack(qry_x), torch.stack(qry_y)
            # form N-way, K-shot task with tensor size [N*W, CHW]
            if verbose:
                print(f'spt_x.size() = {spt_x.size()}')
                print(f'spt_y.size() = {spt_y.size()}')
                print(f'qry_x.size() = {qry_x.size()}')
                print(f'spt_y.size() = {qry_y.size()}')
                print()
            if self.flatten:
                CHW = qry_x.shape[-3:]
                spt_x, spt_y, qry_x, qry_y = spt_x.reshape(-1, *CHW), spt_y.reshape(-1), qry_x.reshape(-1, *CHW), qry_y.reshape(-1)
            ## append to N-way, K-shot task to meta-batch of tasks
            batch_spt_x.append(spt_x); batch_spt_y.append(spt_y)
            batch_qry_x.append(qry_x); batch_qry_y.append(qry_y)
        ## get a meta-set of M N-way, K-way classification tasks [M,K*N,C,H,W]
        batch_spt_x, batch_spt_y, batch_qry_x, batch_qry_y = torch.stack(batch_spt_x), torch.stack(batch_spt_y), torch.stack(batch_qry_x), torch.stack(batch_qry_y)
        return batch_spt_x, batch_spt_y, batch_qry_x, batch_qry_y

def get_meta_set_loader(meta_set, meta_batch_size, n_episodes, n_classes, k_shot, k_eval, pin_mem=True, n_workers=4):
    """[summary]

    Args:
        meta_set ([type]): the meta-set
        meta_batch_size ([type]): [description]
        n_classes ([type]): [description]
        pin_mem (bool, optional): [Since returning cuda tensors in dataloaders is not recommended due to cuda subties with multithreading, instead set pin=True for fast transfering of the data to cuda]. Defaults to True.
        n_workers (int, optional): [description]. Defaults to 4.

    Returns:
        [type]: [description]
    """
    if n_classes > len(meta_set):
        raise ValueError(f'You really want a N larger than the # classes in the meta-set? n_classes, len(meta_set = {n_classes, len(meta_set)}')
    collator_nk_way = GetMetaBatch_NK_WayClassTask(meta_batch_size, n_classes, k_shot, k_eval)
    episodic_sampler = EpisodicSampler(total_classes=len(meta_set), n_episodes=n_episodes)
    episodic_metaloader = data.DataLoader(
        meta_set, 
        num_workers=n_workers, 
        pin_memory=pin_mem, # to make moving to cuda more efficient
        collate_fn=collator_nk_way, # does the collecting to return M N,K-shot task
        batch_sampler=episodic_sampler # for keeping track of the episode
        )
    return episodic_metaloader

def get_meta_set_loaders_miniImagenet(args):
    ## get transforms for images
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train_images = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize])
    transform_val_images = transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize])
    transform_test_images = transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize])

    ## get meta-loaders
    meta_train_set = MetaSet(args.data_root, 'train', args.k_shot, args.k_eval, transform_train_images)
    meta_val_set = MetaSet(args.data_root, 'val', args.k_shot, args.k_eval, transform_val_images)
    meta_test_set = MetaSet(args.data_root, 'test', args.k_shot, args.k_eval, transform_test_images)

    ## get loaders for the meta-sets
    meta_train_loader = get_meta_set_loader(meta_train_set, args.meta_batch_size_train, args.episodes, args.n_classes, args.k_shot, args.k_eval)
    meta_val_loader = get_meta_set_loader(meta_val_set, args.meta_batch_size_eval, args.episodes_val, args.n_classes, args.k_shot, args.k_eval)
    meta_test_loader = get_meta_set_loader(meta_test_set, args.meta_batch_size_eval, args.episodes_test, args.n_classes, args.k_shot, args.k_eval)
    
    return meta_train_loader, meta_val_loader, meta_test_loader

def get_args_for_mini_imagenet():
    from automl.child_models.learner_from_opt_as_few_shot_paper import Learner

    args = SimpleNamespace()
    ##
    args.copy_initial_weights = False
    args.track_higher_grads = True
    ##
    args.n_workers = 4
    # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. 
    # If your data elements are a custom type, or your collate_fn returns a batch that is a custom type, see the example below.
    args.pin_mem = True # it is generally not recommended to return CUDA tensors in multi-process loading because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing (see CUDA in multiprocessing). Instead, we recommend using automatic memory pinning (i.e., setting pin_memory=True), which enables fast data transfer to CUDA-enabled GPUs. https://pytorch.org/docs/stable/data.html
    ##
    args.k_shot = 5
    args.k_eval = 15
    args.n_classes = 5 # N
    args.episodes = 5
    args.episodes_val = 4
    args.episodes_test = 3
    args.image_size = 84 # for img transform
    args.data_root = Path("~/automl-meta-learning/data/miniImagenet").expanduser()
    # careful to have these larger than the size of the meta-set
    args.meta_batch_size_train = 12
    args.meta_batch_size_eval = 10
    ##
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.nb_inner_train_steps = 3
    ##
    args.bn_momentum = 0.95
    args.bn_eps = 1e-3
    args.base_model = Learner(args.image_size, args.bn_eps, args.bn_momentum, args.n_classes)
    ##
    args.criterion = nn.CrossEntropyLoss()
    return args

def test_episodic_loader_inner_loop_per_task_good_accumulator(debug_test=True):
    import torch
    import torch.optim as optim
    from automl.child_models.learner_from_opt_as_few_shot_paper import Learner
    import automl.child_models.learner_from_opt_as_few_shot_paper 
    import higher

    ## get args for test
    args = get_args_for_mini_imagenet()

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

if __name__ == "__main__": 
    import time
    from uutils import report_times

    start = time.time()
    test_episodic_loader_inner_loop_per_task_good_accumulator()
    time_passed_msg, _, _, _ = report_times(start)
    print(time_passed_msg)