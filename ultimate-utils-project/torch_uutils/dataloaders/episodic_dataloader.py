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

    def __init__(self, root, phase='train', n_shot=5, n_eval=15, transform=None):
        """Args:
            root (str): path to data
            phase (str): train, val or test
            n_shot (int): how many examples per class for training (k/n_support)
            n_eval (int): how many examples per class for evaluation
                - n_shot + n_eval = batch_size for data.DataLoader of ClassDataset
            transform (torchvision.transforms): data augmentation
        """
        root = os.path.join(root, phase) # e.g '/Users/brandomiranda/automl-meta-learning/data/miniImagenet/train'
        self.labels = sorted(os.listdir(root))
        
        ## Get data (images)
        # images = [glob.glob(os.path.join(root, label, '*')) for label in self.labels]
        images = []
        for label in self.labels:
            pathname = os.path.join(root, label, '*')
            list_pathnames = glob.glob( pathname ) # Return a possibly-empty list of path names that match pathname
            images.append( list_pathnames )

        ## self.episode_loader = [data.DataLoader( ClassDataset(images=images[idx], label=idx, transform=transform), batch_size=n_shot+n_eval, shuffle=True, num_workers=0) for idx, _ in enumerate(self.labels)]
        self.episode_loader = []
        # loops through each labels/tasks for the meta set split e.g. 64, 16, 20
        for idx, _ in enumerate(self.labels):
            label = idx # class/label/task
            imgs = images[idx] # 600 images, i.e. the sampling from p(x,y|t), this will be split into support & query sets
            classdataset = ClassDataset(images=imgs, label=label, transform=transform) # all 600 images for a specific class/label/task
            taskloader = data.DataLoader(classdataset, batch_size=n_shot+n_eval, shuffle=True, num_workers=0) # data loader for the current class/label/task
            self.episode_loader.append(taskloader)
        print(f'len(self.episode_loader) = {len(self.episode_loader)}')

    def __getitem__(self, idx):
        '''
        Getiitem for EpisodeDataset
        '''
        # sample dataloader from task=idx
        taskloader = self.episode_loader[idx] # dataloader class that samples examples form task, mimics x,y ~ P(x,y|task=idx), tasks are modeled by index/label in this problem
        episode_loader = iter(taskloader)
        # get current data set D = (D^{train},D^{test}) as a [n_shot, c, h, w] tensor
        current_dataset_episode = next(episode_loader) # sample batch of size 20 from 600 available in the current task to for a D = (D^{train},D^{test}) split
        return current_dataset_episode # 5,15 data set split D = (D^{train},D^{test})
        #return next(iter(self.episode_loader[idx]))

    def __len__(self):
        return len(self.labels)


class ClassDataset(data.Dataset):
    '''
    Class that holds all the images for a specific class. So it has the 600 images from class=label.
    '''

    def __init__(self, images, label, transform=None):
        """Args:
            images (list of str): each item is a path to an image of the same label
            label (int): the label of all the images
        """
        self.images = images
        self.label = label
        self.transform = transform

    def __getitem__(self, idx):
        image = PILI.open(self.images[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, self.label

    def __len__(self):
        return len(self.images)


class EpisodicSampler(data.Sampler):

    def __init__(self, total_classes, n_class, n_episode):
        self.total_classes = total_classes
        self.n_class = n_class
        self.n_episode = n_episode

    def __iter__(self):
        for i in range(self.n_episode):
            yield torch.randperm(self.total_classes)[:self.n_class]

    def __len__(self):
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
        ## Forward Pass
        for inner_epoch in range(self.args.nb_inner_train_steps):
                self.args.inner_i = 0
                for batch_idx in range(0, len(inner_inputs), self.args.batch_size):
                    fmodel.train()
                    # get batch for inner training, usually with support/innner set
                    inner_input = inner_inputs[batch_idx:batch_idx+self.args.batch_size].to(self.args.device)
                    inner_target = inner_targets[batch_idx:batch_idx+self.args.batch_size].to(self.args.device)
                    # base/child model forward pass
                    logits = fmodel(inner_input)
                    inner_loss = self.args.criterion(logits, inner_target)
                    inner_train_err = calc_error(mdl=fmodel, X=outer_inputs, Y=outer_targets)
                    # inner-opt update
                    self.add_inner_train_info(diffopt, inner_train_loss=inner_loss, inner_train_err=inner_train_err)
                    if self.inner_debug:
                        self.args.logger.loginfo(f'Inner:[inner_i={self.args.inner_i}], inner_loss: {inner_loss}, inner_train_acc: {inner_train_acc}, test loss: {-1}, test acc: {-1}')
                    self.args.inner_i += 1

        ## Debug statement
        if debug_test:
            print(f"===-->>> outer_debug: phase: EVAL: meta_loss: {meta_loss} outer_train_acc {outer_train_acc}")
 

if __name__ == "__main__":
    test_episodic_loader
