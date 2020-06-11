"""

Notation/terminology:
meta-set = {D_t}_t set of data-set approximating the task distribution. i.e. D_t ~~ p(x,y|t) e.g. |D_t| = 600
D_t = a data-set for a task t. We sample images from there to make the support & query set of size k_shot, k_eval. 
    sampling from D_t is an approximation to sampling from p(x,y|t)

S_t = support set for task t (other notation D^tr_t)
Q_t = query set for task t (other notation D^ts_t))
S_t ~ p^k_shot(x,y|t)
Q_t ~ p^k_eval(x,y|t)

N =  # of classes in N-way
k_shot = # size of support set. |S_t| = k_shot
k_eval = # of query set. |Q_t| = k_eval
M = batch of tasks (sometimes M=N)

B_tasks = indicies of batch of tasks {t_m}^M_m or {D_t}^M_m or {p(x,y|t)}^M_t
S = {S_t}^M_m the split from the sampled data. |S| = sum k_shot_c = K*M when M=N -> |S| = N*K
Q = {Q_t}^M_m the split from the sampled data. usually |Q| = k_eval*M when M=N -> |Q| = N*k_eval
"""

from __future__ import division, print_function, absolute_import

import os
import re
import pdb
import glob
import pickle

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as PILI
import numpy as np

from pathlib import Path

from types import SimpleNamespace

from tqdm import tqdm

def get_support_query_batches(args, episode_x, episode_y):
    """Split the data set of example from the M tasks into it's Support set S and Query set Q.
    
    Note: 
        M is usually N, N=M e.g. 5-way
        S = {S_t}^M_t i.e. all the examples and labels for the batch of tasks in the support sets
        Q = {Q_t}^M_t i.e. all the examples and labels for the batch of tasks in the query sets
    Args:
        args ([type]): [description]
        episode_x ([torch([5, 20, 3, 84, 84])]): data examples from batch of task (k_shot+k_eval for each of the M tasks) e.g. torch.Size([5, 20, 3, 84, 84])
        episode_y ([torch.Size([5, 20])]): [description]

    Returns:
        S_x {tensor([k_shot*N,C,H,W])} - Support set with k_shot image examples for each of the N classes. |S_x| = k_shot*N e.g. torch.Size([25, 3, 84, 84])
        S_y {tensor([k_shot*N])} - Support set with k_shot target examples for each of the N classes. |S_y| = k_shot*N e.g. torch.Size([25])
        Q_x {tensor([k_eval*N,C,H,W])} - Query set with k_eval image examples for each of the N classes. |Q_x| = k_eval*N torch.Size([75, 3, 84, 84])
        Q_y {tensor([k_eval*N])} - Query set with k_eval target examples for each of the N classes. |Q_x| = k_eval*N e.g. torch.Size([75])
    """
    S_x, S_y, Q_x, Q_y = get_inner_outer_batches(args, episode_x, episode_y) 
    return S_x, S_y, Q_x, Q_y

def get_inner_outer_batches(args, episode_x, episode_y):
    """Get the Support & Query sets for meta-training the meta-learner.
    
    Arguments:
        args {Namespace} -- arguments for experiment
        episode_x {TODO} -- Contains the test & train datasets used for meta-trainining.
        episode_y {[type]} -- TODO not used. What is this? Check pytorch code for meta-lstm paper
    
    Returns:
        inner_inputs {TODO} - Train X data-set & bach for inner training.
        inner_targets {TODO} - Train Y data-set & bach for inner training.
        outer_inputs {TODO} -  Test data-set & bach for outer training (or evaluation).
        outer_targets {TODO} - Test data-set & bach for outer training (or evaluation).
    """
    ## Support data sets, D^{train} from current batch of task/classes
    inner_inputs = episode_x[:, :args.k_shot].reshape(-1, *episode_x.shape[-3:]).to(args.device) # [n_class * k_shot, :]
    inner_targets = torch.LongTensor(np.repeat(range(args.n_class), args.k_shot)).to(args.device) # [n_class * k_shot]
    ## Query data sets, D^{test} from current task/classes
    outer_inputs = episode_x[:, args.k_shot:].reshape(-1, *episode_x.shape[-3:]).to(args.device) # [n_class * k_eval, :]
    outer_targets = torch.LongTensor(np.repeat(range(args.n_class), args.k_eval)).to(args.device) # [n_class * k_eval]
    return inner_inputs, inner_targets, outer_inputs, outer_targets

class MetaSet_Dataset(data.Dataset):
    """ Class that models a meta-set {D_t}_t ~~ {p(x,y|t)}_t. e.g. {D_t}_t s.t. t \in {1,...64} for mini-Imagenet's meta-train-set
    Recall that approx. D_t ~~ p(x,y|t) so when sampling from this class t it will give you a set number of examples (k_shot+k_eval)
    that will be used to form the support set S_t and query set Q_t. Note |S_t| = k_shot, |Q_t| = k_eval

    Args:
        data ([type]): [description]
    """

    def __init__(self, root, phase='train', k_shot=5, k_eval=15, transform=None):
        """Args:
            root (str): path to data
            phase (str): train, val or test
            k_shot (int): how many examples per class for training (k/n_support)
            k_eval (int): how many examples per class for evaluation
                - k_shot + k_eval = batch_size for data.DataLoader of ClassDataset
            transform (torchvision.transforms): data augmentation
        """
        ## Locate meta-set & tasl labels/idx i.e. get root to data-sets D_t for each task (where D_t ~ p(x,y|t) through 600 examples)
        root = os.path.join(root, phase) # e.g '/Users/brandomiranda/automl-meta-learning/data/miniImagenet/train'
        # count the number of tasks to create a label for each one
        self.labels = sorted(os.listdir(root)) # e.g. 64 for meta-train, 16 for meta-val, 20 for meta-test
        
        ## Get meta-set {p(x,y|t)}_t ~ {D_t}_t for all tasks e.g. images = list of 64 classes with 600 images.
        # images = [glob.glob(os.path.join(root, label, '*')) for label in self.labels]
        images = [] # e.g. holds all the paths to the images e.g. 64 tasks/classes of mini-imagenet
        for label in self.labels: # for each label get the 600 paths to the images
            ## for each task/label get path to it's approximation to it's distribution p(x,y|t) ~ D_t of 600 images
            # path to D_t approximating p(x,y|t) where 600 images lie. e.g. '/Users/brandomiranda/automl-meta-learning/data/miniImagenet/train/n13133613/*' note * is to match all names of images
            pathname_to_task = os.path.join(root, label, '*') # e.g. path to D_t approximating p(x,y|t) e.g. 600 images for current label/task
            # list of paths to all the 600 jpg images for the current task D_t
            list_pathnames_of_images = glob.glob( pathname_to_task ) # glog.glob = Return a possibly-empty list of path names that match pathname, since at the end we have * we match all images.
            images.append( list_pathnames_of_images ) # append list of pathname to the iamges. e.g. ['/Users/brandomirand...00181.jpg', '/Users/brandomirand...00618.jpg', '/Users/brandomirand...00624.jpg', '/Users/brandomirand...00630.jpg', '/Users/brandomirand...00397.jpg', '/Users/brandomirand...01089.jpg', '/Users/brandomirand...00368.jpg', '/Users/brandomirand...00340.jpg', '/Users/brandomirand...00432.jpg', '/Users/brandomirand...00591.jpg', '/Users/brandomirand...01102.jpg', '/Users/brandomirand...00234.jpg', '/Users/brandomirand...00220.jpg', '/Users/brandomirand...00546.jpg', ...]
        # e.g. images = [D_t]^64_t=1 where |D_t| = 600

        ## Generate list of dataloaders for each task so we can sample k_shot, k_eval examples for that specific task (so this lo)
        # self.episode_loader = [data.DataLoader( ClassDataset(images=images[idx], label=idx, transform=transform), batch_size=k_shot+k_eval, shuffle=True, num_workers=0) for idx, _ in enumerate(self.labels)]
        self.episode_loader = []
        for idx, _ in enumerate(self.labels):
            ##
            # wrap each task D_t ~ p(x,y|t) in a ClassDataset
            label = idx # get class/label for current task e.g. out of 64 tasks idx \in [0,...,63]
            path_to_task_imgs = images[idx] # list for 600 images for the current task
            Dt_classdataset = ClassDataset(images=path_to_task_imgs, label=label, transform=transform) # task D_t
            # wrap dataset so we can sample k_shot, k_eval examples to form the Support and Query set. Note: |S_t| = k_shot, |Q_t| = k_eval
            nb_examples_for_S_Q = k_shot+k_eval
            Dt_taskloader = data.DataLoader(Dt_classdataset, batch_size=nb_examples_for_S_Q, shuffle=True, num_workers=0) # data loader for the current task D_t for class/label t
            self.episode_loader.append(Dt_taskloader) # collect all tasks so this is size e.g. 64 or 16 or 20 for each meta-set
        ## at the end of this loop episode_loader has a list of dataloader's for each task
        ## e.g. approximately episode_loader = {D_t}^64_t=1 and we can sample S_t,Q_t ~ D_t just like we'd do S_t,Q_t ~ p(x,y|t)
        print(f'len(self.episode_loader) = {len(self.episode_loader)}') # e.g. this list is of size 64 for meta-train-set (or 16, 20 meta-val, meta-test)

    def __getitem__(self, idx):
        """Get a batch of examples k_shot+k_eval from a task D_t ~~ p(x,y|t) to be split into the suppport and query set S_t, Q_t.

        Args:
            idx ([int]): index for the task (class label) e.g. idx in range [1 to 64]

        Returns:
            [TODO]: returns the tensor of all examples k_shot+k_eval to be split into S_t and Q_t.
        """
        # return next(iter(self.episode_loader[idx]))
        ## sample dataloader for task D_t
        label = idx # task label e.g. single tensor(22)
        Dt_task_dataloader = iter(self.episode_loader[label]) # dataloader class that samples examples form task, mimics x,y ~ P(x,y|task=idx), tasks are modeled by index/label in this problem
        # get current data set D = (D^{train},D^{test}) as a [k_shot, c, h, w] tensor
        ## Sample k_eval examples to form the query and support set e.g. sample batch of size 20 images from 600 available in the current task to later form S_t,Q_t
        SQ_x,S_y = next(Dt_task_dataloader) # e.g. 20=k_shot+k_eval images and intergers representing the label e.g. (tensor([[[[-0.0801, ....2641]]]]), tensor([22, 22, 22, ...  22, 22]))
        return [SQ_x,S_y] # e.g. tuples of 20 x's and y's for task t/label. All y=t. e.g. (tensor([[[[-0.0801, ....2641]]]]), tensor([22, 22, 22, ...  22, 22]))

    def __len__(self):
        """ Returns the number of tasks D_t.

        Returns:
            [int]: number of tasks e.g. 64 for mini-Imagenet's meta-train-set
        """
        return len(self.labels) # e.g. 64 for mini-Imagenet's meta-train-set 


class ClassDataset(data.Dataset):
    '''
    Class that holds D_t i.e. approximation to the task distribution p(x,y|t) using the 600 examples.
    Class that holds all the images for a specific class/task. So it has the 600 (paths) to the images for each task/class/label.
    It remembers the index for the current task in the label field.
    It remembers the list to all the paths of the 600 images.
    '''

    def __init__(self, images, label, transform=None):
        """Args:
            images (list of str): each item is a path to an image of the same label
            label (int): the label of all the images
        """
        self.images = images # e.g. ['/Users/brandomirand...00106.jpg', '/Users/brandomirand...01218.jpg', '/Users/brandomirand...00660.jpg', '/Users/brandomirand...00674.jpg', '/Users/brandomirand...00884.jpg', '/Users/brandomirand...00890.jpg', '/Users/brandomirand...00648.jpg', '/Users/brandomirand...01230.jpg', '/Users/brandomirand...00338.jpg', '/Users/brandomirand...00489.jpg', '/Users/brandomirand...00270.jpg', '/Users/brandomirand...00264.jpg', '/Users/brandomirand...01152.jpg', '/Users/brandomirand...00258.jpg', ...]
        self.label = label # e.g. 37
        self.transform = transform

    def __getitem__(self, idx):
        """Returns a single example from an approximation to the task distribution p(x,y|t).
        Precisely, it returns (x,y)~D_t where D_t can is of size 600 for mini-imagenet

        Args:
            idx ([int]): an index from D_t (approximating sampling from p(x,y|t) ) e.g. 261

        Returns:
            [tensor, int]: single example (x,y) where y is an int.
        """
        ## get a single image (and transform it if necessary)
        image = PILI.open(self.images[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # return 1 data sample x, y ~ D_t (approximating x,y ~ p(x,y|t) since D_t is of e.g. size 600)
        return image, self.label # real image and it's int label e.g. [3, 84, 84] and int e.g. 37

    def __len__(self):
        """Length of D_t, the data-set approximating the (current) task distribution p(x,y|t).
        Note: D_t ~~ p(x,y|t) approximately

        Returns:
            [int]: number of images x in D_t
        """
        return len(self.images) # e.g. 600


class EpisodicSampler(data.Sampler):
    """ For each episode, sampler a batch of tasks. {t_i}^M_m where t_i is an idx for the task sampled.

    Comments:
        Meant to be passed to batch_sampler when creating a data_loader.
        batch_sampler = A custom Sampler that yields a list of batch indices at a time can be passed as the batch_sampler argument.
    """

    def __init__(self, total_classes, n_class, n_episode):
        """[summary]

        Args:
            total_classes ([int]): total number of tasks for a meta-set e.g. 64, 16, 20 for mini-imagenet
            n_class ([int]): the number of classes to sample e.g. 5
            n_episode ([int]): [description]
        """
        self.total_classes = total_classes # e.g. 64, 16, 20 for mini-imagenet (total number of tasks for a meta-set)
        self.n_class = n_class # the number of classes to sample e.g. 5
        self.n_episode = n_episode # number of times to samples from a task D_t e,g, 60K for MAML/meta-lstm.

    def __iter__(self):
        """Returns/yields a batch of indices representing the batch of tasks.
        If it's the meta-train set with 64 labels and it's and N-way K-shot learning, it returns a list of length N (e.g. 5) with integers ranging in the range 0-63.
        Note, that for more general, this could sample M # of tasks if that's how it's set up for each episode (and have a different number of classes) i.e. N != M in general.

        Yields:
            [list of ints]: yields a batch of tasks (indicies) represented as a list of integers in the range(0,N_meta-set).
        """
        for episode in range(self.n_episode):
            # Sample a random permutation of indices for all tasks/classes e.g. random permutation of nat's from 0 to 63
            random_for_indices_all_tasks = torch.randperm(self.total_classes) # Return a random permutation of integers in a range e.g. tensor([28, 36, 53, 6, 2, 38, 9, 42, 46, 58, 44, 25, 41, 20, 26, 62, 57, 63, 16, 27, 32, 61, 29, 21, 45, 48, 60, 7, 56, 0, 47, 4, 50, 39, 49, 35, 43, 15, 33, 17, 13, 24, 59, 14, 22, 37, 34, 1, 8, 11, 10, 54, 3, 51, 19, 52, 12, 5, 31, 23, 55, 18, 30, 40])
            # Sample a batch of tasks. i.e. select self.n_class (e.g. 5) task indices.
            indices_batch_tasks = random_for_indices_all_tasks[:self.n_class] # e.g. tensor([28, 45, 31, 29,  7])
            # stateful return, when iterator is called it continues executing the next line using the state from the last place it was left off, in this case the main state being remembered is the # of episodes. So this generator ends once the # of episodes has been reached.
            yield indices_batch_tasks

    def __len__(self):
        """Returns the number of episodes.

        Returns:
            [int]: returns the number of times we will sample episodes
        """
        return self.n_episode

def prepare_data_for_few_shot_learning(args):
    """[summary]

    Args:
        args ([type]): [description]

    Returns:
        [DataLoader]: metatrainset_loader; a dataloader that samples the data for a batch of tasks. 
            e.g. a sample of it produces 
        [DataLoader]: metavalset_loader.
        [DataLoader]: metatestset_loader.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ## Get the meta-sets as dataset classes.
    metatrainset = MetaSet_Dataset(args.data_root, 'train', args.k_shot, args.k_eval,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize]))
    metavalset = MetaSet_Dataset(args.data_root, 'val', args.k_shot, args.k_eval,
        transform=transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize]))
    metatestset = MetaSet_Dataset(args.data_root, 'test', args.k_shot, args.k_eval,
        transform=transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize]))
    ## Get episodic samplers. They sampler task (indices) according to args.n_class for args.episodes episodes. e.g. samples 5 tasks (idx) for 60K episodes.
    episode_sampler_metatrainset = EpisodicSampler(len(metatrainset), args.n_class, args.episodes)
    episode_sampler_metavalset = EpisodicSampler(len(metavalset), args.n_class, args.episodes_val)
    episode_sampler_metatestset = EpisodicSampler(len(metatestset), args.n_class, args.episodes_test)
    ## Get the loaders for the meta-sets. These return the sample of tasks used for meta-training.
    metatrainset_loader = data.DataLoader(metatrainset, num_workers=args.n_workers, pin_memory=args.pin_mem, batch_sampler=episode_sampler_metatrainset)
    metavalset_loader = data.DataLoader(metavalset, num_workers=4, pin_memory=False, batch_sampler=episode_sampler_metavalset)
    metatestset_loader = data.DataLoader(metatestset, num_workers=4, pin_memory=False, batch_sampler=episode_sampler_metatestset)
    return metatrainset_loader, metavalset_loader, metatestset_loader

def get_args_for_mini_imagenet():
    from automl.child_models.learner_from_opt_as_few_shot_paper import Learner

    args = SimpleNamespace()
    #
    args.n_workers = 4
    args.pin_mem = True
    #
    args.k_shot = 5
    args.k_eval = 15
    args.n_class = 5 # M, # of tasks to sample. Note N=M, N from N way
    args.episodes = 5
    args.episodes_val = 4
    args.episodes_test = 3
    args.image_size = 84
    args.data_root = Path("~/automl-meta-learning/data/miniImagenet").expanduser()
    args.batch_size = 25
    #
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.nb_inner_train_steps = 2
    #
    args.bn_momentum = 0.95
    args.bn_eps = 1e-3
    args.base_model = Learner(args.image_size, args.bn_eps, args.bn_momentum, args.n_class)
    return args

def test_episodic_loader(debug_test=True):
    args = get_args_for_mini_imagenet()
    metatrainset_loader, metavalset_loader, metatestset_loader = prepare_data_for_few_shot_learning(args)
    ##
    for episode, (SQ_x, SQ_y) in enumerate(metatrainset_loader): # samples of data from batch of tasks to form S & Q e.g. SQ_x.size() = torch.Size([5, 20, 3, 84, 84]) SQ_y.size() = torch.Size([5, 20])
        print(f'episode/outer_i = {episode}')
        ## Split the features and target labels into the support S and query Q sets for the batch of M tasks
        S_x, S_y, Q_x, Q_x = get_support_query_batches(args, SQ_x, SQ_y)
        print()
        ## Forward Pass
        for inner_epoch in range(self.args.nb_inner_train_steps):
                for batch_idx in range(0, len(inner_inputs), self.args.batch_size):
                    print()
 

if __name__ == "__main__":
    test_episodic_loader()
