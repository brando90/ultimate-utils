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
    return get_inner_outer_batches(args, episode_x, episode_y) 

def get_inner_outer_batches(args, episode_x, episode_y):
    """Get the data sets for training the meta learner. Recall that for the Meta-Train-Set we get the
    train data and test data for the current task and use both to train the meta-learner.
    Thus this function samples a task and gets the pair of train & test sets used to train meta-learner.
    
    Arguments:
        args {Namespace} -- arguments for experiment
        episode_x {TODO} -- Contains the test & train datasets used for meta-trainining.
        episode_y {[type]} -- TODO not used. What is this? Check pytorch code for meta-lstm paper
    
    Returns:
        inner_inputs {TODO} - Train X data-set & bach for inner training.
        inner_targets {TODO} - Train Y data-set & bach for inner training.
        outer_inputs {TODO} -  Test data-set & bach for outer training (or evaluation).
        outer_targets {TODO} - Test data-set & bach for outer training (or evaluation)..
    """
    ## Support data sets, D^{train} from current batch of task/classes
    inner_inputs = episode_x[:, :args.n_shot].reshape(-1, *episode_x.shape[-3:]).to(args.device) # [n_class * n_shot, :]
    inner_targets = torch.LongTensor(np.repeat(range(args.n_class), args.n_shot)).to(args.device) # [n_class * n_shot]
    ## Query data sets, D^{test} from current task/classes
    outer_inputs = episode_x[:, args.n_shot:].reshape(-1, *episode_x.shape[-3:]).to(args.device) # [n_class * n_eval, :]
    outer_targets = torch.LongTensor(np.repeat(range(args.n_class), args.n_eval)).to(args.device) # [n_class * n_eval]
    return inner_inputs, inner_targets, outer_inputs, outer_targets

class EpisodeDataset(data.Dataset):
    """ Class that models a meta-set {D_t}_t ~~ {p(x,y|t)}_t. e.g. {D_t}_t s.t. t \in {1,...64} for mini-Imagenet's meta-train-set
    Recall that approx. D_t ~~ p(x,y|t) so when sampling from this class t it will give you a set number of examples (k_shot+k_eval)
    that will be used to form the support set S_t and query set Q_t. Note |S_t| = k_shot, |Q_t| = k_eval

    Args:
        data ([type]): [description]
    """

    def __init__(self, root, phase='train', n_shot=5, n_eval=15, transform=None):
        """Args:
            root (str): path to data
            phase (str): train, val or test
            n_shot (int): how many examples per class for training (k/n_support)
            n_eval (int): how many examples per class for evaluation
                - n_shot + n_eval = batch_size for data.DataLoader of ClassDataset
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
        # self.episode_loader = [data.DataLoader( ClassDataset(images=images[idx], label=idx, transform=transform), batch_size=n_shot+n_eval, shuffle=True, num_workers=0) for idx, _ in enumerate(self.labels)]
        self.episode_loader = []
        for idx, _ in enumerate(self.labels):
            ##
            # wrap each task D_t ~ p(x,y|t) in a ClassDataset
            label = idx # get class/label for current task e.g. out of 64 tasks idx \in [0,...,63]
            path_to_task_imgs = images[idx] # list for 600 images for the current task
            Dt_classdataset = ClassDataset(images=path_to_task_imgs, label=label, transform=transform) # task D_t
            # wrap dataset so we can sample k_shot, k_eval examples to form the Support and Query set. Note: |S_t| = k_shot, |Q_t| = k_eval
            nb_examples_for_S_Q = n_shot+n_eval
            Dt_taskloader = data.DataLoader(Dt_classdataset, batch_size=nb_examples_for_S_Q, shuffle=True, num_workers=0) # data loader for the current task D_t for class/label t
            self.episode_loader.append(Dt_taskloader) # collect all tasks so this is size e.g. 64 or 16 or 20 for each meta-set
        ## at the end of this loop episode_loader has a list of dataloader's for each task
        ## e.g. approximately episode_loader = {D_t}^64_t=1 and we can sample S_t,Q_t ~ D_t just like we'd do S_t,Q_t ~ p(x,y|t)
        print(f'len(self.episode_loader) = {len(self.episode_loader)}') # e.g. this list is of size 64 for meta-train-set (or 16, 20 meta-val, meta-test)

    def __getitem__(self, idx):
        """Get a batch of examples n_shot+n_eval from a task D_t ~~ p(x,y|t) to be split into the suppport and query set S_t, Q_t.

        Args:
            idx ([int]): index for the task (class label) e.g. idx in range [1 to 64]

        Returns:
            [TODO]: returns the tensor of all examples n_shot+n_eval to be split into S_t and Q_t.
        """
        # return next(iter(self.episode_loader[idx]))
        ## sample dataloader for task D_t
        label = idx # task label e.g. single tensor(22)
        Dt_task_dataloader = iter(self.episode_loader[label]) # dataloader class that samples examples form task, mimics x,y ~ P(x,y|task=idx), tasks are modeled by index/label in this problem
        # get current data set D = (D^{train},D^{test}) as a [n_shot, c, h, w] tensor
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
    """

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
        for i in range(self.n_episode):
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
    '''
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # get the meta-sets
    train_set = EpisodeDataset(args.data_root, 'train', args.n_shot, args.n_eval,
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
    val_set = EpisodeDataset(args.data_root, 'val', args.n_shot, args.n_eval,
        transform=transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize]))
    test_set = EpisodeDataset(args.data_root, 'test', args.n_shot, args.n_eval,
        transform=transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize]))
    # get the loaders for the meta-sets
    trainset_loader = data.DataLoader(train_set, num_workers=args.n_workers, pin_memory=args.pin_mem,
        batch_sampler=EpisodicSampler(len(train_set), args.n_class, args.episodes))
    valset_loader = data.DataLoader(val_set, num_workers=2, pin_memory=False,
        batch_sampler=EpisodicSampler(len(val_set), args.n_class, args.episodes_val))
    testset_loader = data.DataLoader(test_set, num_workers=2, pin_memory=False,
        batch_sampler=EpisodicSampler(len(test_set), args.n_class, args.episodes_test))
    return trainset_loader, valset_loader, testset_loader

def get_args_for_mini_imagenet():
    from automl.child_models.learner_from_opt_as_few_shot_paper import Learner

    args = SimpleNamespace()
    #
    args.n_workers = 4
    args.pin_mem = True
    #
    args.n_shot = 5
    args.n_eval = 15
    args.n_class = 5
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
    trainset_loader, valset_loader, testset_loader = prepare_data_for_few_shot_learning(args)
    ##
    for outer_i, (episode_x, episode_y) in enumerate(trainset_loader):
        ## Get batch of tasks and the corresponding Support,Query = D^{train},D^{test} data-sets
        inner_inputs, inner_targets, outer_inputs, outer_targets = get_inner_outer_batches(args, episode_x, episode_y)
        print()
        ## Forward Pass
        for inner_epoch in range(self.args.nb_inner_train_steps):
                for batch_idx in range(0, len(inner_inputs), self.args.batch_size):
                    print()
 

if __name__ == "__main__":
    test_episodic_loader()
