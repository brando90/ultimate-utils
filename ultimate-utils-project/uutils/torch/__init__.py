'''
Torch Based Utils/universal methods

Utils class with useful helper functions

utils: https://www.quora.com/What-do-utils-files-tend-to-be-in-computer-programming-documentation

'''
from datetime import datetime
from typing import List

import higher
import torch
import torch.nn as nn

import numpy as np
import scipy.integrate as integrate
import pandas as pd

from collections import OrderedDict

import dill as pickle

import os

from pathlib import Path

import copy

from pdb import set_trace as st

# from sklearn.linear_model import logistic
from scipy.stats import logistic
from torch.multiprocessing import Pool

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def hello():
    print('hello')

def helloworld():
    print('hello world torch_utils!')

# meta-optimizer utils

def set_requires_grad(bool, mdl):
    for name, w in mdl.named_parameters():
        w.requires_grad = bool


def check_mdl_in_single_gpu(mdl):
    """
    note this only checks the first param and from that infers the rest is in gpu.

    https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180
    :return:
    """
    device = next(mdl.parameters()).device
    return device

def create_detached_deep_copy_old(mdl):
    mdl_new = copy.deepcopy(mdl)
    detached_params = mdl.state_dict()
    # set to detached
    for name, w in mdl.named_parameters():
        w_detached = nn.Parameter(w.detach())
        detached_params[name] = w_detached
    # load model
    mdl_new.load_state_dict(detached_params)
    return mdl_new

def create_detached_deep_copy(human_mdl, mdl_to_copy):
    '''
    create a deep detached copy of mdl_new.
    Needs the human_mdl (instantiated by a human) as an empty vessel and then
    copy the parameters from the real model we want (mdl_to_copy) and returns a filled in
    copy of the human_mdl.
    Essentially does:
    empty_vessel_mdl = deep_copy(human_mdl)
    mdl_new.fill() <- copy(from=mdl_to_copy,to=human_mdl)
    '''
    empty_vessel_mdl = copy.deepcopy(human_mdl)
    # set to detached
    detached_params = empty_vessel_mdl.state_dict()
    for name, w in mdl_to_copy.named_parameters():
        w_detached = nn.Parameter(w.detach())
        detached_params[name] = w_detached
    # load model
    empty_vessel_mdl.load_state_dict(detached_params)
    mdl_new = empty_vessel_mdl
    return mdl_new


def _create_detached_copy_old(mdl, deep_copy, requires_grad):
    """
    DOES NOT WORK. NEED TO FIX. one needs to use modules instead of parameters
    Creates a detached copy (shallow or deep) of the given mode. The given model will have
    its own gradients and form it's own computation tree.
    `
    Arguments:
        mdl {[type]} -- neural net
        deep_copy {bool} -- flag for deep or shallow copy
        requires_grad {bool} -- indicates if to collect gradients
    
    Returns:
        [type] -- detached copy of neural net
    """
    raise ValueError('Does not work')
    new_params = []
    for name, w in mdl.named_parameters():
        # create copy
        if deep_copy:
            w_new = w.clone().detach()
        else:
            w_new = w.detach()
        #w_new = nn.Parameter(w_new)
        # set requires_grad
        w_new.requires_grad = requires_grad
        # append
        new_params.append((name, w_new))
    # create new model
    mdl_new = nn.Sequential(OrderedDict(new_params))
    return mdl_new


# LSTM utils

def get_init_hidden(batch_size, hidden_size, nb_layers, bidirectional, device=None):
    """
    Args:
        batch_size: (int) size of batch
        hidden_size:
        n_layers:
        bidirectional: (torch.Tensor) initial hidden state (n_layers*nb_directions, batch_size, hidden_size)

    Returns:
        hidden:

    Gets initial hidden states for all cells depending on # batches, nb_layers, directions, etc.

    Details:
    We have to have a hidden initial state of size (hidden_size) for:
    - each sequence in the X_batch
    - each direction the RNN process the sequence
    - each layer of the RNN (note we are stacking RNNs not mini-NN layers)

    NOTE: notice that we don't have seq_len anywhere because the first hidden
    state is only needed to start the computation

    :param int batch_size: size of batch
    :return torch.Tensor hidden: initial hidden state (n_layers*nb_directions, batch_size, hidden_size)
    """
    # get gpu
    use_cuda = torch.cuda.is_available()
    device_gpu_if_avail = torch.device("cuda" if use_cuda else "cpu")
    device = device if device==None else device_gpu_if_avail
    ## get initial memory and hidden cell (c and h)
    nb_directions = 2 if bidirectional else 1
    h_n = torch.randn(nb_layers * nb_directions, batch_size, hidden_size, device=device)
    c_n = torch.randn(nb_layers * nb_directions, batch_size, hidden_size, device=device)
    hidden = (h_n, c_n)
    return hidden

def lp_norm(mdl, p=2):
    lp_norms = [w.norm(p) for name, w in mdl.named_parameters()]
    return sum(lp_norms)

def lp_norm_grads(mdl, p=2, grads=False):
    lp_norms = [w.grad.norm(p) for name, w in mdl.named_parameters()]
    return sum(lp_norms)

def check_two_models_equal(model1, model2):
    '''
    Checks if two models are equal.

    https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351
    '''
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        # if p1.data.ne(p2.data).sum() > 0:
        if (p1 != p2).any():
            return False
    return True

def are_all_params_leafs(mdl):
    all_leafs = True
    for (name, w) in mdl.named_parameters():
        all_leafs = all_leafs and w.is_leaf
    return all_leafs

def calc_error(mdl: torch.nn.Module, X: torch.Tensor, Y):
    # acc == (true != mdl(x).max(1).item() / true.size(0)
    train_acc = calc_accuracy(mdl, X, Y)
    train_err = 1.0 - train_acc
    return train_err

def calc_accuracy(mdl: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Get the accuracy with respect to the most likely label.

    ref: https://stackoverflow.com/questions/51503851/calculate-the-accuracy-every-epoch-in-pytorch/63271002#63271002

    :param mdl:
    :param X:
    :param Y:
    :return:
    """
    # get the scores for each class (or logits)
    y_logits = mdl(X)  # unnormalized probs
    # -- return the values & indices with the largest value in the dimension where the scores for each class is
    # get the scores with largest values & their corresponding idx (so the class that is most likely)
    max_scores, max_idx_class = y_logits.max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
    # usually 0th coordinate is batch size
    n = X.size(0)
    assert(n == max_idx_class.size(0))
    # -- calulate acc (note .item() to do float division)
    acc = (max_idx_class == Y).sum() / n
    return acc.item()

def calc_accuracy_from_logits(y_logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Returns accuracy between tensors
    :param y_logits:
    :param y:
    :return:
    """
    max_logits, max_indices_classes = y_logits.max(dim=1)  # [B, C] -> [B]
    n_examples = y.size(0)  # usually baatch_size
    assert(n_examples == max_indices_classes.size(0))
    acc = (max_indices_classes == y).sum() / n_examples
    return acc.item()

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    print("WARNING UNTESTED code")
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(-1, 1).expand_as(y_pred)  # [B] -> [B, 1] -> [B, maxk]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [B, maxk] were for each example we know which topk prediction matched truth

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:, :k]  # [B, maxk] -> [B, maxk]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

def accuracy_original(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

    return res

def get_stats(flatten_tensor):
    """Get some stats from tensor.
    
    Arguments:
        flatten_tensor {torchTensor} -- torch tensor to get stats
    
    Returns:
        [list torch.Tensor] -- [mu, std, min_v, max_v, med]
    """
    mu, std = flatten_tensor.mean(), flatten_tensor.std()
    min_v, max_v, med = flatten_tensor.min(), flatten_tensor.max(), flatten_tensor.median()
    return [mu, std, min_v, max_v, med]

def add_inner_train_info_simple(diffopt, *args, **kwargs):
    """  Function that adds any train info desired to be passed to the diffopt to be used during the inner update step.

    Arguments:
        diffopt {trainable optimizer} -- trainable optimizer.
    """
    diffopt.param_groups[0]['kwargs']['trainfo_kwargs'] = kwargs
    diffopt.param_groups[0]['kwargs']['trainfo_args'] = args

def add_inner_train_stats(diffopt, *args, **kwargs):
    """ Add any train info desired to pass to diffopt for it to use during the update step.
    
    Arguments:
        diffopt {trainable optimizer} -- trainable optimizer.
    """
    inner_loss = kwargs['inner_loss']
    inner_train_err = kwargs['inner_train_err']
    diffopt.param_groups[0]['kwargs']['prev_trainable_opt_state']['train_loss'] = inner_loss
    diffopt.param_groups[0]['kwargs']['prev_trainable_opt_state']['inner_train_err'] = inner_train_err

####

def save_ckpt_meta_learning(args, meta_learner, debug=False):
    # https://discuss.pytorch.org/t/advantages-disadvantages-of-using-pickle-module-to-save-models-vs-torch-save/79016
    # make dir to logs (and ckpts) if not present. Throw no exceptions if it already exists
    path_to_ckpt = args.logger.current_logs_path
    path_to_ckpt.mkdir(parents=True, exist_ok=True) # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.
    ckpt_path_plus_path = path_to_ckpt / Path('db')

    # Pickle args & logger (note logger is inside args already), source: https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
    db = {} # database dict
    tb = args.tb
    args.tb = None
    args.base_model = "no child mdl in args see meta_learner" # so that we don't save the child model so many times since it's part of the meta-learner
    db['args'] = args # note this obj has the last episode/outer_i we ran
    db['meta_learner'] = meta_learner
    torch.save(db, ckpt_path_plus_path)
    # with open(ckpt_path_plus_path , 'wb+') as db_file:
    #     pickle.dump(db, db_file)
    if debug:
        test_ckpt_meta_learning(args, meta_learner, debug)
    args.base_model = meta_learner.base_model # need to re-set it otherwise later in the code the pointer to child model will be updated and code won't work
    args.tb = tb
    return

def save_checkpoint_simple(args, meta_learner):
    # make dir to logs (and ckpts) if not present. Throw no exceptions if it already exists
    args.path_2_save_ckpt.mkdir(parents=True, exist_ok=True)  # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.

    args.base_model = "check the meta_learner field in the checkpoint not in the args field"  # so that we don't save the child model so many times since it's part of the meta-learner
    # note this obj has the last episode/outer_i we ran
    torch.save({'args': args, 'meta_learner': meta_learner}, args.path_2_save_ckpt / Path('/ckpt_file'))

def resume_ckpt_meta_learning(args):
    path_to_ckpt = args.resume_ckpt_path / Path('db')
    with open(path_to_ckpt, 'rb') as db_file: 
        db = pickle.load(db_file)
        args_recovered = db['args']
        meta_learner = db['meta_learner']
        args_recovered.base_model = meta_learner.base_model
        # combine new args with old args
        args.base_model = "no child mdl in args see meta_learner"
        args = args_recovered
        return args, meta_learner

def test_ckpt_meta_learning(args, meta_learner, verbose=False):
    path_to_ckpt = args.logger.current_logs_path
    path_to_ckpt.mkdir(parents=True, exist_ok=True) # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.
    ckpt_path_plus_path = path_to_ckpt / Path('db')

    # Pickle args & logger (note logger is inside args already), source: https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
    db = {} # database dict
    args.base_model = "no child mdl" # so that we don't save the child model so many times since it's part of the meta-learner
    db['args'] = args # note this obj has the last episode/outer_i we ran
    args.base_model = meta_learner.base_model # need to re-set it otherwise later in the code the pointer to child model will be updated and code won't work
    db['meta_learner'] = meta_learner
    with open(ckpt_path_plus_path , 'wb+') as db_file:
        dumped_outer_i = args.outer_i
        pickle.dump(db, db_file)
    with open(ckpt_path_plus_path , 'rb') as db_file:
        args = get_args_debug(path=path_to_ckpt)
        loaded_outer_i = args.outer_i
    if verbose:
        print(f'==> dumped_outer_i = {dumped_outer_i}')
        print(f'==> loaded_outer_i = {loaded_outer_i}')
    ## Assertion Tests
    assert(dumped_outer_i == loaded_outer_i)
    return


def test_ckpt(args, verbose=False):
    path_to_ckpt = args.logger.current_logs_path
    path_to_ckpt.mkdir(parents=True, exist_ok=True) # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.
    ckpt_path_plus_path = path_to_ckpt / Path('db')

    ## Pickle args & logger (note logger is inside args already), source: https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
    db = {} # database dict
    db['args'] = args # note this obj has the last episode/outer_i we ran
    with open(ckpt_path_plus_path , 'wb+') as db_file:
        dumped_outer_i = args.outer_i
        pickle.dump(db, db_file)
    with open(ckpt_path_plus_path , 'rb') as db_file:
        args = get_args_debug(path=path_to_ckpt)
        loaded_outer_i = args.outer_i
    if verbose:
        print(f'==> dumped_outer_i = {dumped_outer_i}')
        print(f'==> loaded_outer_i = {loaded_outer_i}')
    ## Assertion Tests
    assert(dumped_outer_i == loaded_outer_i)
    return

def get_args_debug(args=None, path=''):
    if args is not None:
        path_to_ckpt = args.resume_ckpt_path / Path('db')
    else:
        path_to_ckpt = path / Path('db')
    ## open db file
    db_file = open(path_to_ckpt, 'rb')
    db = pickle.load(db_file)
    args = db['args']
    db_file.close()
    return args

def resume_ckpt_meta_lstm(metalearner, optim, resume, device):
    ckpt = torch.load(resume, map_location=device)
    last_episode = ckpt['episode']
    metalearner.load_state_dict(ckpt['metalearner'])
    optim.load_state_dict(ckpt['optim'])
    return last_episode, metalearner, optim

def save_ckpt_meta_lstm(episode, metalearner, optim, save):
    if not os.path.exists(os.path.join(save, 'ckpts')):
        os.mkdir(os.path.join(save, 'ckpts'))

    torch.save({
        'episode': episode,
        'metalearner': metalearner.state_dict(),
        'optim': optim.state_dict()
    }, os.path.join(save, 'ckpts', 'meta-learner-{}.pth.tar'.format(episode)))


def resume_ckpt_meta_lstm(metalearner, optim, resume, device):
    ckpt = torch.load(resume, map_location=device)
    last_episode = ckpt['episode']
    metalearner.load_state_dict(ckpt['metalearner'])
    optim.load_state_dict(ckpt['optim'])
    return last_episode, metalearner, optim

##

def replace_bn(module, name):
    """
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    """
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            new_bn = torch.nn.BatchNorm2d(target_attr.num_features, target_attr.eps, target_attr.momentum, target_attr.affine,
                                          track_running_stats=False)
            setattr(module, attr_str, new_bn)

    # "recurse" iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_bn(immediate_child_module, name)

def set_tracking_running_stats(model):
    """
    https://discuss.pytorch.org/t/batchnorm1d-with-batchsize-1/52136/8
    https://stackoverflow.com/questions/64920715/how-to-use-have-batch-norm-not-forget-batch-statistics-it-just-used-in-pytorch

    @param model:
    @return:
    """
    for attr in dir(model):
        if 'bn' in attr:
            target_attr = getattr(model, attr)
            target_attr.track_running_stats = True
            target_attr.running_mean = torch.nn.Parameter(torch.zeros(target_attr.num_features, requires_grad=False))
            target_attr.running_var = torch.nn.Parameter(torch.ones(target_attr.num_features, requires_grad=False))
            target_attr.num_batches_tracked = torch.nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)
            # target_attr.reset_running_stats()
    return

##

def count_nb_params(net):
    count = 0
    for p in net.parameters():
        count += p.data.nelement()
    return count

##

def gradient_clip(args, meta_opt):
    """Do gradient clipping: * If ‖g‖ ≥ c Then g := c * g/‖g‖

    depending on args it does it per parameter or all parameters together.
    
    Arguments:
        args {Namespace} -- arguments for experiment
        meta_opt {Optimizer} -- optimizer that train the meta-learner
    
    Raises:
        ValueError: For invalid arguments to args.grad_clip_mode
    """
    #do gradient clipping: * If ‖g‖ ≥ c Then g := c * g/‖g‖
    # note: grad_clip_rate is a number for clipping the other is the type
    # of clipping we are doing
    if args.grad_clip_rate is not None:
        if args.grad_clip_mode == 'clip_all_seperately':
            for group_idx, group in enumerate(meta_opt.param_groups):
                for p_idx, p in enumerate(group['params']):
                    nn.utils.clip_grad_norm_(p, args.grad_clip_rate)
        elif args.grad_clip_mode == 'clip_all_together':
            # [y for x in list_of_lists for y in x] 
            all_params = [ p for group in meta_opt.param_groups for p in group['params'] ]
            nn.utils.clip_grad_norm_(all_params, args.grad_clip_rate)
        elif args.grad_clip_mode == 'no_grad_clip' or args.grad_clip_mode is None: # i.e. do not clip if grad_clip_rate is None
            pass
        else:
            raise ValueError(f'Invalid, args.grad_clip_mode = {args.grad_clip_mode}')

def preprocess_grad_loss(x, p=10, eps=1e-8):
    """ Preprocessing (vectorized) implementation from the paper:

    if |x| >= e^-p (not too small)
        coord1, coord2 = (log(|x| + eps)/p, sign(x))
    else: (too small
        coord1, coord2 = (-1, (e^p)*x)
    return stack(coord1,coord2)
    
    usually applied to loss and grads.

    Arguments:
        x {[torch.Tensor]} -- input to preprocess
    
    Keyword Arguments:
        p {int} -- number that indicates the scaling (default: {10})
        eps {float} - numerical stability param (default: {1e-8})
    
    Returns:
        [torch.Tensor] -- preprocessed numbers
    """
    if len(x.size()) == 0:
        x = x.unsqueeze(0)
    # implements vectorized if statement
    indicator = (x.abs() >= np.exp(-p)).to(torch.float32)

    # preproc1 - magnitude path (coord 1) log(|x|)/(p+eps) or -1
    # if not too small use the exponent of the magnitude/p
    # if too small use a -1 to indicate too small to the neural net
    x_proc1 = indicator * torch.log(x.abs() + eps) / p + (1 - indicator) * -1
    # preproc2 - sign path (coord 2) sign(x) or (e^p)*x
    # if not too small log(|x|)/p
    # if too small (e^p)*x
    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
    # stack
    # usually in meta-lstm x is n_learner_params so this forms a tensor of size [n_learnaer_params, 2]
    x_proc = torch.stack([x_proc1, x_proc2], 1)
    return x_proc

#

def functional_diff_norm(f1, f2, lb=-1.0, ub=1.0, p=2):
    """
    Computes norm:

    ||f||_p = (int_S |f|^p dmu)^1/p

    https://en.wikipedia.org/wiki/Lp_space

    https://stackoverflow.com/questions/63237199/how-does-one-compute-the-norm-of-a-function-in-python
    """
    # index is there since it also returns acc/err
    if 'torch' in str(type(f1)) or 'torch' in str(type(f2)):
        pointwise_diff = lambda x: abs(f1(torch.tensor([x])) - f2(torch.tensor([x]))) ** p
    else:
        pointwise_diff = lambda x: abs(f1(x) - f2(x)) ** p
    norm, abs_err = integrate.quad(pointwise_diff, a=lb, b=ub)
    return norm**(1/p), abs_err

def cxa_dist(mdl1, mdl2, meta_batch, layer_name, cca_size=None, iters=2, cxa_dist_type='pwcca'):
    # print(cca_size)
    # meta_batch [T, N*K, CHW], [T, K, D]
    from anatome import SimilarityHook
    # get sim/dis functions
    hook1 = SimilarityHook(mdl1, layer_name, cxa_dist_type)
    hook2 = SimilarityHook(mdl2, layer_name, cxa_dist_type)
    mdl1.eval()
    mdl2.eval()
    for _ in range(iters):  # might make sense to go through multiple is NN is stochastic e.g. BN, dropout layers
        # x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
        # x = torch.torch.distributions.Uniform(low=-1, high=1).sample((15, 1))
        # x = torch.torch.distributions.Uniform(low=-1, high=1).sample((500, 1))
        x = meta_batch
        mdl1(x)
        mdl2(x)
    # dist = hook1.distance(hook2, size=cca_size, cca_distance='pwcca')
    dist = hook1.distance(hook2, size=cca_size)
    # if cxa_dist_type == 'lincka':
    #     sim = dist  # only true for cka for this library
    #     dist = 1 - sim  # since dist = 1 - sim but cka is outputing in this library the sim
    return dist

def cxa_sim(mdl1, mdl2, meta_batch, layer_name, cca_size=None, iters=2, cxa_sim_type='pwcca'):
    dist = cxa_dist(mdl1, mdl2, meta_batch, layer_name, cca_size, iters, cxa_sim_type)
    return 1 - dist

def cca_rand_data(mdl1, mdl2, num_samples_per_task, layer_name, lb=-1, ub=1, Din=1, cca_size=None, iters=2):
    # meta_batch [T, N*K, CHW], [T, K, D]
    from anatome import SimilarityHook
    # get sim/dis functions
    hook1 = SimilarityHook(mdl1, layer_name)
    hook2 = SimilarityHook(mdl2, layer_name)
    mdl1.eval()
    mdl2.eval()
    for _ in range(iters):  # might make sense to go through multiple is NN is stochastic e.g. BN, dropout layers
        x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
        # x = torch.torch.distributions.Uniform(low=-1, high=1).sample((15, 1))
        # x = torch.torch.distributions.Uniform(low=-1, high=1).sample((num_samples_per_task, 1))
        mdl1(x)
        mdl2(x)
    dist = hook1.distance(hook2, size=cca_size)
    return dist

def ned(f, y):
    """
    Normalized euncleadian distance

    ned = sqrt 0.5*np.var(x - y) / (np.var(x) + np.var(y)) = 0.5 variance of difference / total variance individually

    reference: https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance

    @param x:
    @param y:
    @return:
    """
    ned = ( 0.5*np.var(f - y) / (np.var(f) + np.var(y)) )**0.5
    return ned

def r2_score_from_torch(y_true: torch.Tensor, y_pred: torch.Tensor):
    """ returns the accuracy from torch tensors """
    from sklearn.metrics import r2_score
    acc = r2_score(y_true=y_true.detach().numpy(), y_pred=y_pred.detach().numpy())
    return acc

def r2_symmetric(f, y, r2_type='explained_variance'):
    """
    Normalized (symmetric) R^2 with respect to two vectors:

        check if statements for equation.

    reference:
    - https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance
    - https://en.wikipedia.org/wiki/Coefficient_of_determination#:~:text=R2%20is%20a%20statistic,predictions%20perfectly%20fit%20the%20data.
    - https://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score
    - https://en.wikipedia.org/wiki/Fraction_of_variance_unexplained
    - https://en.wikipedia.org/wiki/Explained_variation
    - https://en.wikipedia.org/wiki/Mahalanobis_distance

    @param x:
    @param y:
    @return:
    """
    # import sklearn.metrics.explained_variance_score as evar
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import r2_score

    f = f if type(f) != torch.Tensor else f.detach().cpu().numpy()
    y = y if type(y) != torch.Tensor else y.detach().cpu().numpy()
    if r2_type == 'average_r2s':
        r2_f = r2_score(y_true=f, y_pred=y)
        r2_y = r2_score(y_true=y, y_pred=f)
        r2 = 0.5*r2_f + 0.5*r2_y
    elif r2_type == 'normalized_average_r2s':
        r2_f = r2_score(y_true=f, y_pred=y)
        r2_y = r2_score(y_true=y, y_pred=f)
        r2 = 0.5 * r2_f + 0.5 * r2_y
        # sig = torch.nn.Sigmoid()
        # r2 = sig(r2).item()
        raise ValueError(f'Not implemented {r2_type}')
    elif r2_type == 'mohalanobis':
        # https://en.wikipedia.org/wiki/Mahalanobis_distance
        from scipy.spatial import distance
        # xy = np.vstack((f.T,y.T))
        # S = np.cov(xy)
        # r2 = distance.mahalanobis(f.squeeze(), y.squeeze(), S)
        raise ValueError(f'Not implemented: {r2_type}')
    elif r2_type == '1_minus_total_residuals':
        # not using this anymore, gave weird results
        # r2 = 1 - ((2 * mse(f, y)) / (np.var(f) + np.var(y)))
        r2 = 1 - ((mse(f, y)) / (np.var(f) + np.var(y)))
    elif r2_type == 'ned':
        r2 = ned(f, y)
    elif r2_type == 'cosine':
        raise ValueError(f'Not implemented {r2_type}')
    elif r2_type == 'my_explained_variance':
        # evar_f2y
        # evar_y2f
        # r2 = (evar_f2y + evar_y2f) / (np.var(f) + np.var(y))
        raise ValueError(f'Not implemented: {r2_type}')
    else:
        raise ValueError(f'Not implemented: {r2_type}')
    return r2

def normalized_r2_old(y_true, y_pred, normalizer='Sigmoid'):
    """

    :param normalizer: Sigmoid otherwise use Tanh
    :param y_true:
    :param y_pred:
    :return:
    """
    from sklearn.metrics import r2_score
    from scipy.stats import logistic

    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    if normalizer == 'Sigmoid':
        # norm = logistic.cdf
        # norm_r2 = (1.0/norm(1.0))*norm(r2)
        norm = logistic.cdf
        norm_r2 = (1.0/norm(1.0))*norm(r2)
    else:
        raise ValueError(f'Normalizer {normalizer} not implemented')
    return norm_r2

def normalized_r2(y_true, y_pred, normalizer='Sigmoid'):
    """

    :param normalizer: Sigmoid otherwise use Tanh
    :param y_true:
    :param y_pred:
    :return:
    """
    import np
    from sklearn.metrics import r2_score
    from scipy.stats import logistic

    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    if normalizer == 'Sigmoid':
        if r2 > 0:
            norm_r2 = 0.5*r2 + 0.5
        else:
            norm_r2 = logistic.cdf(r2)
    elif normalizer == 'tanh':
        if r2 > 0:
            norm_r2 = r2
        else:
            norm_r2 = np.tanh(r2)
    else:
        raise ValueError(f'Normalizer {normalizer} not implemented')
    return norm_r2

# def normalized_r2_torch(y_true, y_pred, normalizer='Sigmoid'):
#     """
#
#     :param normalizer: Sigmoid otherwise use Tanh
#     :param y_true:
#     :param y_pred:
#     :return:
#     """
#     from sklearn.metrics import r2_score
#     from scipy.stats import logistic
#
#     # y_true=qry_y_t.detach().numpy(), y_pred=qry_logits_t.detach().numpy()
#     sig = torch.nn.Sigmoid() if normalizer == 'Sigmoid' else torch.nn.Tanh()
#     r2_score = ignite.contrib.metrics.regression.R2Score()
#     # r2 = r2_score(y_true=y_true, y_pred=y_pred)
#     norm_r2 = logistic(r2).item()
#     return norm_r2

# def cca(mdl1, mdl2, meta_batch, layer_name, cca_size=None, iters=2):
#     # meta_batch [T, N*K, CHW], [T, K, D]
#     from anatome import SimilarityHook
#     # get sim/dis functions
#     hook1 = SimilarityHook(mdl1, layer_name)
#     hook2 = SimilarityHook(mdl2, layer_name)
#     for _ in range(iters):  # might make sense to go through multiple is NN is stochastic e.g. BN, dropout layers
#         x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
#         mdl1(x)
#         mdl2(x)
#     dist = hook1.distance(hook2, size=cca_size)
#     return dist

# def cca(mdl1, mdl2, dataloader, cca_size=None, iters=10):
#     # with torch.no_grad()
#     for _ in range(iters):
#         next()
#         mdl1(x)
#         mdl2(x)

def l2_sim_torch(x1, x2, dim=1, sim_type='nes_torch'):
    if sim_type == 'nes_torch':
        sim = nes_torch(x1, x2, dim)
    elif sim_type == 'cosine_torch':
        cos = nn.CosineSimilarity(dim=dim)
        sim = cos(x1, x2)
    else:
        raise ValueError(f'Not implemented sim_type={sim_type}')
    return sim

def ned_torch(x1: torch.Tensor, x2: torch.Tensor, dim=1, eps=1e-8) -> torch.Tensor:
    """
    Normalized eucledian distance in pytorch.

    Cases:
        1. For comparison of two vecs directly make sure vecs are of size [B] e.g. when using nes as a loss function.
            in this case each number is not considered a representation but a number and B is the entire vector to
            compare x1 and x2.
        2. For comparison of two batch of representation of size 1D (e.g. scores) make sure it's of shape [B, 1].
            In this case each number *is* the representation of the example. Thus a collection of reps
            [B, 1] is mapped to a rep of size [B]. Note usually D does decrease since reps are not of size 1
            (see case 3)
        3. For the rest specify the dimension. Common use case [B, D] -> [B, 1] for comparing two set of
            activations of size D. In the case when D=1 then we have [B, 1] -> [B, 1]. If you meant x1, x2 [D, 1] to be
            two vectors of size D to be compare feed them with shape [D].
            This one is also good for computing the NED for two batches of values. e.g. if you have a tensor of size
            [B, k] and the row is a batch and each entry is the y value for that batch. If a batch is a task then
            it is computing the NED for that task, which is good because that batch has it's own unique scale that
            we are trying to normalize and by doing it per task you are normalizing it as you'd expect (i.e. per task).

    Note: you cannot use this to compare two single numbers NED(x,y) is undefined because a single number does not have
    a variance. Variance[x] = undefined.

    https://discuss.pytorch.org/t/how-does-one-compute-the-normalized-euclidean-distance-similarity-in-a-numerically-stable-way-in-a-vectorized-way-in-pytorch/110829
    https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance/498753?noredirect=1#comment937825_498753
    https://github.com/brando90/Normalized-Euclidean-Distance-and-Similarity
    """
    # to compute ned for two individual vectors e.g to compute a loss (NOT BATCHES/COLLECTIONS of vectorsc)
    if len(x1.size()) == 1:
        # [K] -> [1]
        ned_2 = 0.5 * ((x1 - x2).var() / (x1.var() + x2.var() + eps))
    # if the input is a (row) vector e.g. when comparing two batches of acts of D=1 like with scores right before sf
    elif x1.size() == torch.Size([x1.size(0), 1]):  # note this special case is needed since var over dim=1 is nan (1 value has no variance).
        # [B, 1] -> [B]
        ned_2 = 0.5 * ((x1 - x2)**2 / (x1**2 + x2**2 + eps)).squeeze()  # Squeeze important to be consistent with .var, otherwise tensors of different sizes come out without the user expecting it
    # common case is if input is a batch
    else:
        # e.g. [B, D] -> [B]
        ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5

def nes_torch(x1, x2, dim=1, eps=1e-8):
    return 1 - ned_torch(x1, x2, dim, eps)


def tensorify(lst):
    """
    List must be nested list of tensors (with no varying lengths within a dimension).
    Nested list of nested lengths [D1, D2, ... DN] -> tensor([D1, D2, ..., DN)

    :return: nested list D
    """
    # base case, if the current list is not nested anymore, make it into tensor
    if type(lst) != list:
        # if it's a float or a tensor already (the single element)
        return torch.tensor(lst)
    if type(lst[0]) != list:
        if type(lst) == torch.Tensor:
            return lst
        elif type(lst[0]) == torch.Tensor:
            return torch.stack(lst, dim=0)
        else:  # if the elements of lst are floats or something like that
            return torch.tensor(lst)
    # recursive case, for every sub list get it into tensor (recursively) form and then combine with torch.stack
    current_dimension_i = len(lst)
    for d_i in range(current_dimension_i):
        tensor = tensorify(lst[d_i])
        lst[d_i] = tensor
    # end of loop lst[d_i] = tensor([D_i, ... D_0])
    tensor_lst = torch.stack(lst, dim=0)
    return tensor_lst

def floatify_results(dic):
    if type(dic) is not dict:
        if type(dic) is torch.Tensor:
            if len(dic.size()) == 1:
                lst_floats = [val.item() for val in dic]
                return lst_floats
            elif len(dic.size()) == 0:
                return dic.squeeze().item()
            else:
                raise ValueError(f'Invalid value: {dic}')
    elif type(dic) is None:
        return dic
    else:
        d = {}
        for k, v in dic.items():
            d[k] = floatify_results(v)
        return d


def print_results_old(args, all_meta_eval_losses, all_diffs_qry,  all_diffs_cca, all_diffs_cka, all_diffs_neds):
    print(f'experiment {args.data_path}\n')

    print(f'Meta Val loss (using query set of course, (k_val = {args.k_eval}))')
    meta_val_loss_mean = np.average(all_meta_eval_losses)
    meta_val_loss_std = np.std(all_meta_eval_losses)
    print(f'-> meta_val_loss = {meta_val_loss_mean} +-{meta_val_loss_std}')

    print(f'\nFuntional difference according to query set, (approx integral with k_val = {args.k_eval})')
    diff_qry_mean = np.average(all_diffs_qry)
    diff_qry_std = np.std(all_diffs_qry)
    print(f'-> diff_qrt_mean = {diff_qry_mean} +-{diff_qry_std}')

    print(f'Funtional difference according to cca (k_val = {args.k_eval})')
    diff_cca_mean = np.average(all_diffs_cca)
    diff_cca_std = np.std(all_diffs_cca)
    print(f'-> diff_cca_mean = {diff_cca_mean} +-{diff_cca_std}')

    print(f'Funtional difference according to cka (k_val = {args.k_eval})')
    diff_cka_mean = np.average(all_diffs_cka)
    diff_cka_std = np.std(all_diffs_cka)
    print(f'-> diff_cka_mean = {diff_cka_mean} +-{diff_cka_std}')

    # print(f'Funtional difference according to cka (k_val = {args.k_eval})')
    # diff_cka_mean = np.average(all_diffs_cka)
    # diff_cka_std = np.std(all_diffs_cka)
    # print(f'-> diff_cca_mean = {diff_cka_mean} +-{diff_cka_std}')

    print(f'Funtional difference according to ned (k_val = {args.k_eval})')
    diff_ned_mean = np.average(all_diffs_neds)
    diff_ned_std = np.std(all_diffs_neds)
    print(f'-> diff_ned_mean = {diff_ned_mean} +-{diff_ned_std}')

    # print(f'Funtional difference according to r2s_avg (k_val = {args.k_eval})')
    # diff_r2avg_mean = np.average(all_diffs_r2_avg)
    # diff_r2avg_std = np.std(all_diffs_r2_avg)
    # print(f'-> diff_r2avg_mean = {diff_r2avg_mean} +-{diff_r2avg_std}')

    # print(f'Funtional difference according to integral approx')
    # diff_approx_int_mean = np.average(all_diffs_approx_int)
    # diff_approx_int_std = np.std(all_diffs_approx_int)
    # print(f'-> diff_qrt_mean = {diff_approx_int_mean} +-{diff_approx_int_std}')

    # print(f'Funtional difference according to r2_1_mse_var')
    # diff_r2_1_mse_var_mean = np.average(all_diffs_r2_1_mse_var)
    # diff_r2_1_mse_var_std = np.std(all_diffs_r2_1_mse_var)
    # print(f'-> diff_ned_mean = {diff_r2_1_mse_var_mean} +-{diff_r2_1_mse_var_std}')


def print_results(args, all_meta_eval_losses, all_diffs_qry,  all_diffs_cca, all_diffs_cka, all_diffs_neds):
    print(f'experiment {args.data_path}\n')

    print(f'Meta Val loss (using query set of course, (k_val = {args.k_eval}))')
    meta_val_loss_mean = np.average(all_meta_eval_losses)
    meta_val_loss_std = np.std(all_meta_eval_losses)
    print(f'-> meta_val_loss = {meta_val_loss_mean} +-{meta_val_loss_std}')

    print(f'\nFuntional difference according to query set, (approx integral with k_val = {args.k_eval})')
    diff_qry_mean = np.average(all_diffs_qry)
    diff_qry_std = np.std(all_diffs_qry)
    print(f'-> diff_qrt_mean = {diff_qry_mean} +-{diff_qry_std}')

    print(f'Funtional difference according to cca (k_val = {args.k_eval})')
    diff_cca_mean = np.average(all_diffs_cca)
    diff_cca_std = np.std(all_diffs_cca)
    print(f'-> diff_cca_mean = {diff_cca_mean} +-{diff_cca_std}')

    print(f'Funtional difference according to cka (k_val = {args.k_eval})')
    diff_cka_mean = np.average(all_diffs_cka)
    diff_cka_std = np.std(all_diffs_cka)
    print(f'-> diff_cka_mean = {diff_cka_mean} +-{diff_cka_std}')

    # print(f'Funtional difference according to cka (k_val = {args.k_eval})')
    # diff_cka_mean = np.average(all_diffs_cka)
    # diff_cka_std = np.std(all_diffs_cka)
    # print(f'-> diff_cca_mean = {diff_cka_mean} +-{diff_cka_std}')

    print(f'Funtional difference according to ned (k_val = {args.k_eval})')
    diff_ned_mean = np.average(all_diffs_neds)
    diff_ned_std = np.std(all_diffs_neds)
    print(f'-> diff_ned_mean = {diff_ned_mean} +-{diff_ned_std}')

    # print(f'Funtional difference according to r2s_avg (k_val = {args.k_eval})')
    # diff_r2avg_mean = np.average(all_diffs_r2_avg)
    # diff_r2avg_std = np.std(all_diffs_r2_avg)
    # print(f'-> diff_r2avg_mean = {diff_r2avg_mean} +-{diff_r2avg_std}')

    # print(f'Funtional difference according to integral approx')
    # diff_approx_int_mean = np.average(all_diffs_approx_int)
    # diff_approx_int_std = np.std(all_diffs_approx_int)
    # print(f'-> diff_qrt_mean = {diff_approx_int_mean} +-{diff_approx_int_std}')

    # print(f'Funtional difference according to r2_1_mse_var')
    # diff_r2_1_mse_var_mean = np.average(all_diffs_r2_1_mse_var)
    # diff_r2_1_mse_var_std = np.std(all_diffs_r2_1_mse_var)
    # print(f'-> diff_ned_mean = {diff_r2_1_mse_var_mean} +-{diff_r2_1_mse_var_std}')

def compute_result_stats(all_sims):
    cxas = ['cca', 'cka']
    l2 = ['nes', 'cosine']
    stats = {metric: {'avg': None, 'std': None, 'rep': {'avg': None, 'std': None}, 'all': {'avg': None, 'std': None}} for metric, _ in all_sims.items()}
    for metric, tensor_of_metrics in all_sims.items():
        if metric in cxas:
            # compute average cxa per layer: [T, L] -> [L]
            avg_sims = tensor_of_metrics.mean(dim=0)
            std_sims = tensor_of_metrics.std(dim=0)
            # compute representation & all avg cxa [T, L] -> [1]
            L = tensor_of_metrics.size(1)
            indicies = torch.tensor(range(L-1))
            representation_tensors = tensor_of_metrics.index_select(dim=1, index=indicies)
            avg_sims_representation_layer = representation_tensors.mean()
            std_sims_representation_layer = representation_tensors.std()

            avg_sims_all = tensor_of_metrics.mean()
            std_sims_all = tensor_of_metrics.std()
        elif metric in l2:
            # compute average l2 per layer: [T, L, K_eval] -> [L]
            avg_sims = tensor_of_metrics.mean(dim=[0, 2])
            std_sims = tensor_of_metrics.std(dim=[0, 2])
            # compute representation & all avg l2 [T, L, K_eval] -> [1]
            L = tensor_of_metrics.size(1)
            indicies = torch.tensor(range(L-1))
            representation_tensors = tensor_of_metrics.index_select(dim=1, index=indicies)
            avg_sims_representation_layer = representation_tensors.mean()
            std_sims_representation_layer = representation_tensors.std()

            avg_sims_all = tensor_of_metrics.mean()
            std_sims_all = tensor_of_metrics.std()
        else:
            # compute average [T] -> [1]
            avg_sims = tensor_of_metrics.mean(dim=0)
            std_sims = tensor_of_metrics.std(dim=0)
        stats[metric]['avg'] = avg_sims
        stats[metric]['std'] = std_sims
        if metric in cxas+l2:
            stats[metric]['rep']['avg'] = avg_sims_representation_layer
            stats[metric]['rep']['std'] = std_sims_representation_layer
            stats[metric]['all']['avg'] = avg_sims_all
            stats[metric]['all']['std'] = std_sims_all
    return stats

def get_mean_std_pairs(metric: dict):
    """

    :param metric: dict with avg & std keys as keys mapping to list or floats
    e.g.
        "cca": {
        ...,
        "avg": [0.6032, 0.5599, 0.4918, 0.4044],
        ...,
        "std": [0.0297, 0.0362, 0.0948, 0.2481]
    }
    :return:

    TODO: doing significant figures properly
    """
    values_in_columns = []
    if type(metric['avg']) == list:
        paired = zip(metric['avg'], metric['std'])
    else:
        paired = [(metric['avg'], metric['std'])]
    # make output
    # sep = '$\pm$'
    sep = '+-'
    for avg, std in paired:
        values_in_columns.append(f'{avg:.3f}{sep}{std:.3f}')
    return values_in_columns

# -- similarity comparisons for MAML

def parallel_functional_similarities(self, spt_x, spt_y, qry_x, qry_y, layer_names, iter_tasks=None):
    """
    :return: sims = dictionary of metrics of tensors
    sims = {
        (cxa) cca, cka = tensor([I, L]),
        (l2) nes, cosine = tensor([I, L, K_eval])
        nes_output = tensor([I])
        }

    Important note:
    -When a Tensor is sent to another process, the Tensor data is shared. If torch.Tensor.grad is not None,
        it is also shared.
    - After a Tensor without a torch.Tensor.grad field is sent to the other process,
        it creates a standard process-specific .grad Tensor that is not automatically shared across all processes,
        unlike how the Tensor’s data has been shared.
        - this is good! that way when different tasks are being adapted with MAML, their gradients don't "crash"
    - above from docs on note box: https://pytorch.org/docs/stable/notes/multiprocessing.html

    Note:
        - you probably do not have to call share_memory() but it's probably safe to do so. Since docs say the following:
            Moves the underlying storage to shared memory. This is a no-op if the underlying storage is already in
            shared memory and for CUDA tensors. Tensors in shared memory cannot be resized.
            https://pytorch.org/docs/stable/tensors.html#torch.Tensor.share_memory_

    To increase file descriptors
        ulimit -Sn unlimited
    """
    # to have shared memory tensors
    self.base_model.share_memory()
    # this is needed so that each process has access to the layer names needed to compute sims, a bit ugly oh well.
    self.layer_names = layer_names

    # compute similarities in parallel
    # spt_x.share_memory_(), spt_y.share_memory_(), qry_x.share_memory_(), qry_y.share_memory_()
    meta_batch_size = spt_x.size(0)
    iter_tasks = meta_batch_size if iter_tasks is None else min(iter_tasks, meta_batch_size)
    batch_of_tasks = [(spt_x[t], spt_y[t], qry_x[t], qry_y[t]) for t in range(iter_tasks)]

    # num_procs = iter_tasks
    torch.multiprocessing.set_sharing_strategy('file_system')
    # num_procs = 8
    num_procs = torch.multiprocessing.cpu_count() - 2
    print(f'num_procs in pool: {num_procs}')
    with Pool(num_procs) as pool:
        sims_from_pool = pool.map(self.compute_sim_for_current_task, batch_of_tasks)

    # sims is an T by L lists of lists, i.e. each row corresponds to similarities for a specific task for all layers
    sims = {'cca': [], 'cka': [], 'nes': [], 'cosine': [], 'nes_output': [], 'query_loss': []}
    for similiarities_single_dict in sims_from_pool:  # for each result from the multiprocessing result, place in the right sims place
        for metric, s in similiarities_single_dict.items():
            sims[metric].append(s)

    # convert everything to torch tensors
    # sims = {k: tensorify(v) for k, v in sims.items()}
    similarities = {}
    for k, v in sims.items():
        print(f'{k}')
        similarities[k] = tensorify(v)
    return similarities

def compute_sim_for_current_task(self, task):
    # print(f'start: {torch.multiprocessing.current_process()}')
    # unpack args pased via checking global variables, oh no!
    layer_names = self.layer_names
    inner_opt = self.inner_opt
    # unpack args
    spt_x_t, spt_y_t, qry_x_t, qry_y_t = task
    # compute sims
    with higher.innerloop_ctx(self.base_model, inner_opt, copy_initial_weights=self.args.copy_initial_weights,
                              track_higher_grads=self.args.track_higher_grads) as (fmodel, diffopt):
        diffopt.fo = self.fo
        for i_inner in range(self.args.nb_inner_train_steps):
            fmodel.train()
            # base/child model forward pass
            spt_logits_t = fmodel(spt_x_t)
            inner_loss = self.args.criterion(spt_logits_t, spt_y_t)
            # inner-opt update
            diffopt.step(inner_loss)

        # get similarities cca, cka, nes, cosine and nes_output
        fmodel.eval()
        self.base_model.eval()

        # get CCA & CKA per layer (T by 1 by L)
        x = qry_x_t
        if torch.cuda.is_available():
            x = x.cuda()
        cca = self.get_cxa_similarities_per_layer(self.base_model, fmodel, x, layer_names,
                                                  sim_type='pwcca')  # 1 by T
        cka = self.get_cxa_similarities_per_layer(self.base_model, fmodel, x, layer_names,
                                                  sim_type='lincka')  # 1 by T

        # get l2 sims per layer (T by L by k_eval)
        nes = self.get_l2_similarities_per_layer(self.base_model, fmodel, qry_x_t, layer_names,
                                                 sim_type='nes_torch')
        cosine = self.get_l2_similarities_per_layer(self.base_model, fmodel, qry_x_t, layer_names,
                                                    sim_type='cosine_torch')

        # (T by 1)
        y = self.base_model(qry_x_t)
        y_adapt = fmodel(qry_x_t)
        # if not torch.cuda.is_available():
        #     y, y_adapt = y.cpu().detach().numpy(), y_adapt.cpu().detach().numpy()
        # dim=0 because we have single numbers and we are taking the NES in the batch direction
        nes_output = torch.nes_torch(y.squeeze(), y_adapt.squeeze(), dim=0).item()

        query_loss = self.args.criterion(y, y_adapt).item()
    # print(f'done: {torch.multiprocessing.current_process()}')
    # sims = [cca, cka, nes, cosine, nes_output, query_loss]
    sims = {'cca': cca, 'cka': cka, 'nes': nes, 'cosine': cosine, 'nes_output': nes_output, 'query_loss': query_loss}
    # sims = [sim.detach() for sim in sims]
    sims = {metric: tensorify(sim).detach() for metric, sim in sims.items()}
    return sims

# -- distance comparisons for SL

def get_distance_of_inits(args, batch, f1, f2):
    layer_names = args.layer_names
    batch_x, batch_y = batch
    # get CCA & CKA per layer (T by 1 by L)
    cca = get_cxa_distances_per_layer(f1, f2, batch_x, layer_names, dist_type='pwcca')  # 1 by T
    cka = get_cxa_distances_per_layer(f1, f1, batch_x, layer_names, dist_type='lincka')  # 1 by T

    # get l2 sims per layer (T by L by k_eval)
    # nes = self.get_l2_similarities_per_layer(f1, fmodel, qry_x_t, layer_names,
    #                                          sim_type='nes_torch')
    # cosine = self.get_l2_similarities_per_layer(f1, fmodel, qry_x_t, layer_names,
    #                                             sim_type='cosine_torch')
    #
    # # (T by 1)
    # y = f1(qry_x_t)
    # y_adapt = fmodel(qry_x_t)
    # # if not torch.cuda.is_available():
    # #     y, y_adapt = y.cpu().detach().numpy(), y_adapt.cpu().detach().numpy()
    # # dim=0 because we have single numbers and we are taking the NES in the batch direction
    # nes_output = torch.nes_torch(y.squeeze(), y_adapt.squeeze(), dim=0).item()
    #
    # query_loss = self.args.criterion(y, y_adapt).item()
    # print(f'done: {torch.multiprocessing.current_process()}')
    # sims = [cca, cka, nes, cosine, nes_output, query_loss]
    sims = {'cca': cca, 'cka': cka, 'nes': nes, 'cosine': cosine, 'nes_output': nes_output, 'query_loss': query_loss}
    # sims = [sim.detach() for sim in sims]
    sims = {metric: tensorify(sim).detach() for metric, sim in sims.items()}
    return sims

def get_cxa_distances_per_layer(mdl1, mdl2, X, layer_names, dist_type='pwcca'):
    # get [..., s_l, ...] cca sim per layer (for this data set)
    from uutils.torch import cxa_sim

    sims_per_layer = []
    for layer_name in layer_names:
        # sim = cxa_sim(mdl1, mdl2, X, layer_name, cca_size=self.args.cca_size, iters=1, cxa_sim_type=sim_type)
        sim = cxa_dist(mdl1, mdl2, X, layer_name, iters=1, cxa_sim_type=sim_type)
        sims_per_layer.append(sim)
    return sims_per_layer  # [..., s_l, ...]_l

###### MISC

# def log_validation(args, meta_learner, outer_opt, meta_val_set, check_point=True):
#     """ Log the validation loss, acc. Checkpoint the model if that flag is on. """
#     if check_point:
#         store_check_point(args, meta_learner)
#     acc_mean, acc_std, loss_mean, loss_std = meta_eval(args, meta_learner, meta_val_set, iter_limit=args.eval_iters)
#     if acc_mean > args.best_acc:
#         args.best_acc, args.best_loss = acc_mean, loss_mean
#         # args.logger.loginfo(f"***> Stats of Best Acc model: meta-val loss: {args.loss_of_best} +- {loss_std}, meta-val acc: {args.best_acc} +- {acc_std}")
#         if check_point and acc_mean > 0:  # accs < 0 means its regression not really classificaiton
#             store_check_point(args, meta_learner, f'ckpt_file_best_loss.pt')
#     if loss_mean < args.best_loss:
#         args.best_loss = loss_mean
#         if check_point:
#             store_check_point(args, meta_learner, f'ckpt_file_best_loss.pt')
#     return acc_mean, acc_std, loss_mean, loss_std


#######

def test_ned():
    import torch.nn as nn

    # dim = 1  # apply cosine accross the second dimension/feature dimension

    k = 4  # number of examples
    d = 8  # dimension of feature space
    for d in range(1, d):
        x1 = torch.randn(k, d)
        x2 = x1 * 3
        print(f'x1 = {x1.size()}')
        ned_tensor = ned_torch(x1, x2)
        print(ned_tensor)
        print(ned_tensor.size())
        #print(ned_torch(x1, x2, dim=dim))

def test_tensorify():
    t = [1, 2, 3]
    print(tensorify(t).size())
    tt = [t, t, t]
    print(tensorify(tt))
    ttt = [tt, tt, tt]
    print(tensorify(ttt))

def test_normalized_r2():
    # from sklearn.linear_model import logistic
    from scipy.stats import logistic
    def norm_r2(r2):
        return (1.0 / logistic.cdf(1.0)) * logistic.cdf(r2)
    normalized_r2 = norm_r2

    normalized_r2(-1000)
    print(f'{normalized_r2(-1000)=}')

    print(f'{normalized_r2(0)}')

    print(f'{normalized_r2(1)}')

def test_topk_accuracy_and_accuracy():
    import torch
    import torch.nn as nn

    in_features = 1
    n_classes = 10
    batch_size = 4

    mdl = nn.Linear(in_features=in_features, out_features=n_classes)

    x = torch.randn(batch_size, in_features)
    y_logits = mdl(x)
    y = torch.randint(high=n_classes, size=(batch_size,))

    acc_top1, acc_top2, acc_top5 = accuracy(output=y_logits, target=y, topk=(1, 2, 5))
    acc_top1_, acc_top2_, acc_top5_ = topk_accuracy(output=y_logits, target=y, topk=(1, 2, 5))
    assert(acc_top5 == acc_top5_)
    assert(acc_top1 == acc_top1_)
    acc1 = calc_accuracy(mdl, x, y)
    acc1_ = calc_accuracy_from_logits(y_logits, y)
    assert(acc1 == acc1_)
    assert(acc1_ == acc_top1)

if __name__ == '__main__':
    # test_ned()
    # test_tensorify()
    # test_normalized_r2()
    test_topk_accuracy_and_accuracy()
    print('Done\a')
