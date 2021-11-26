"""
Torch Based Utils/universal methods

Utils class with useful helper functions

utils: https://www.quora.com/What-do-utils-files-tend-to-be-in-computer-programming-documentation

todo:
    - write a abstract/general method for training
        1. with iteration based (with progress bars) + batches
        2. with epoch based with progress bars + batches, nested progress bars
        3. fitting one batch
"""
import dill
import gc
from datetime import datetime
from typing import List, Union, Any, Optional, Iterable

import torch
from torch import Tensor, optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.integrate as integrate
import pandas as pd

from collections import OrderedDict

import dill as pickle

import os

from pathlib import Path

import copy

from argparse import Namespace

from uutils.torch_uu.tensorboard import log_2_tb

import gc

import urllib.request

from pprint import pprint

from pdb import set_trace as st

# uutils
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Batch = list


def hello():
    """

    On the terminal do:
    python -c "import uutils; uutils.torch_uu.hello()"
    """
    import uutils.torch_uu as torch_uu
    print(f'\nhello from torch_uu __init__.py in:\n{torch_uu}\n')


def gpu_test_torch_any_device():
    """
    python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"
    """
    from torch import Tensor

    print(f'device name: {device_name()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x: Tensor = torch.randn(2, 4).to(device)
    y: Tensor = torch.randn(4, 1).to(device)
    out: Tensor = (x @ y)
    assert out.size() == torch.Size([2, 1])
    print(f'Success, torch works with whatever device is shown in the output tensor:\n{out=}')


def gpu_test():
    """
    python -c "import uutils; uutils.torch_uu.gpu_test()"
    """
    from torch import Tensor

    print(f'device name: {device_name()}')
    x: Tensor = torch.randn(2, 4).cuda()
    y: Tensor = torch.randn(4, 1).cuda()
    out: Tensor = (x @ y)
    assert out.size() == torch.Size([2, 1])
    print(f'Success, no Cuda errors means it worked see:\n{out=}')


# -

def device_name():
    return gpu_name_otherwise_cpu()


def gpu_name_otherwise_cpu():
    gpu_name_or_cpu = None
    try:
        gpu_name_or_cpu = torch.cuda.get_device_name(0)
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_name_or_cpu = device
    return gpu_name_or_cpu


def make_code_deterministic(seed: int, always_use_deterministic_algorithms: bool = True):
    """

    Note: use_deterministic_algorithms makes all algorithms deterministic while torch_uu.backends.cudnn.deterministic=True
    makes convs only determinsitic. There is also a way to choose algorithms based on hardware performance, to
    avoid that use torch_uu.backends.cudnn.benchmark = False (note the agorithm chosen even if determinsitic might be
    random itself so the other two flags are useful).

    todo -
     - fix this:
      RuntimeError: Deterministic behavior was enabled with either `torch_uu.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
     - figure out the worker in dataloader thing...https://pytorch.org/docs/stable/notes/randomness.html
     - read the RNN LSTM part
    ref:
        - https://pytorch.org/docs/stable/notes/randomness.html
        - https://stackoverflow.com/questions/66130547/what-does-the-difference-between-torch-backends-cudnn-deterministic-true-and
    :return:
    """
    import random
    import numpy as np
    import torch
    # - make pytorch determinsitc
    # makes all ops determinsitic no matter what. Note this throws an errors if you code has an op that doesn't have determinsitic implementation
    torch.manual_seed(seed)
    if always_use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    # makes convs deterministic
    torch.backends.cudnn.deterministic = True
    # doesn't allow benchmarking to select fastest algorithms for specific ops
    torch.backends.cudnn.benchmark = False
    # - make python determinsitic
    np.random.seed(seed)
    random.seed(seed)


def index(tensor: Tensor, value, ith_match: int = 0) -> Union[int, Tensor]:
    """
    Returns generalized index (i.e. location/coordinate) of the first occurence of value
    in Tensor. For flat tensors (i.e. arrays/lists) it returns the indices of the occurrences
    of the value you are looking for. Otherwise, it returns the "index" as a coordinate.
    If there are multiple occurences then you need to choose which one you want with ith_index.
    e.g. ith_index=0 gives first occurence.

    Reference: https://stackoverflow.com/a/67175757/1601580
    :return:
    """
    # bool tensor of where value occurred
    places_where_value_occurs = (tensor == value)
    # get matches as a "coordinate list" where occurence happened
    matches = (tensor == value).nonzero()  # [number_of_matches, tensor_dimension]
    if matches.size(0) == 0:  # no matches
        return -1
    else:
        # get index/coordinate of the occurence you want (e.g. 1st occurence ith_match=0)
        index = matches[ith_match]
        return index


def insert_unk(vocab):
    """
    Inserts unknown token into torchtext vocab.
    """
    from torchtext.vocab import Vocab

    # undos throwing erros when out of vocab is attempted
    default_index = -1
    vocab.set_default_index(default_index)
    assert vocab['out of vocab'] == -1
    # insert unk token
    unk_token = '<unk>'
    if unk_token not in vocab:
        vocab.insert_token(unk_token, 0)
        assert vocab['<unk>'] == 0
    # make default index same as index of unk_token
    vocab.set_default_index(vocab[unk_token])
    assert vocab['out of vocab'] is vocab[unk_token]
    assert isinstance(vocab, Vocab)
    return vocab


def insert_special_symbols(vocab):
    from torchtext.vocab import Vocab

    special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']
    for i, special_symbol in enumerate(special_symbols):
        vocab.insert_token(special_symbol, i)
        if special_symbol == '<unk>':
            vocab.set_default_index(i)
            assert i == 0
        # assert special_symbol not in vocab, f'Dont add the {special_symbol=} when calling this function.'
    assert vocab['out of vocab'] is vocab['<unk>']
    assert vocab['<pad>'] == 1
    assert vocab['<sos>'] == 2
    assert vocab['<eos>'] == 3
    assert isinstance(vocab, Vocab)
    return vocab


def diagonal_mask(size: int, device) -> Tensor:
    """
    Returns the additive diagonal where first entry is zero so that SOS is not removed
    and the remaining diagonal is -inf so that the transformer decoder doesn't cheat.

    ref: https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask/68396781#68396781
    e.g.
    tensor([[0., -inf, -inf, -inf],
        [0., 0., -inf, -inf],
        [0., 0., 0., -inf],
        [0., 0., 0., 0.]])
    :param size:
    :param device:
    :return:
    """
    # returns the upper True diagonal matrix where the diagonal is True also (thats why transpose is needed)
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    # wherever there is a 0 put a -inf
    mask = mask.float().masked_fill(mask == 0, float('-inf'))
    # wherever there is a 1 put a 0
    mask = mask.masked_fill(mask == 1, float(0.0))
    # to device
    mask = mask.to(device)
    return mask


def process_batch_simple(args: Namespace, x_batch, y_batch):
    if isinstance(x_batch, Tensor):
        x_batch = x_batch.to(args.device)
    if isinstance(y_batch, Tensor):
        y_batch = y_batch.to(args.device)
    return x_batch, y_batch


def process_meta_batch(args, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # - upack the data
    if type(batch) == dict:
        (spt_x, spt_y), (qry_x, qry_y) = batch["train"], batch["test"]
    elif type(batch) == tuple or type(batch) == list:
        spt_x, spt_y, qry_x, qry_y = batch
    else:
        raise ValueError(f'Not implemented how to process this batch of type {type(batch)} with value {batch=}')
    # invariant: we have spt_x, spt_y, qry_x, qry_y after here

    # - convert to float32 single float, somehow the ckpts seem to need this for sinusoid
    if hasattr(args, 'to_single_float_float32'):
        if args.to_single_float_float32:
            spt_x, spt_y, qry_x, qry_y = spt_x.to(torch.float32), spt_y.to(torch.float32), qry_x.to(
                torch.float32), qry_y.to(torch.float32)
    return spt_x.to(args.device), spt_y.to(args.device), qry_x.to(args.device), qry_y.to(args.device)


# def get_model(mdl: Union[nn.Module, DistributedDataParallel]) -> nn.Module:
#     if isinstance(mdl, DistributedDataParallel):
#         return mdl.module
#     else:
#         return mdl

def set_requires_grad(bool, mdl):
    for name, w in mdl.named_parameters():
        w.requires_grad = bool


def print_dict_of_dataloaders_dataset_types(dataloaders):
    msg = 'dataset/loader type: '
    for split, dataloader in dataloaders.items():
        dataset = dataloader.dataset
        msg += f'{split=}: {dataset=}, '
    print(msg)


def print_dataloaders_info(opts, dataloaders, split):
    print(f'{split=}')
    print(f'{dataloaders[split]=}')
    print(f'{type(dataloaders[split])=}')
    print(f"{len(dataloaders[split].dataset)=}")
    print(f"{len(dataloaders[split])=}")
    print(f"{len(dataloaders[split].dataset)//opts.batch_size=}")
    if hasattr(opts, 'world_size'):
        print(f"{len(dataloaders[split])*opts.world_size=}")


def check_mdl_in_single_gpu(mdl):
    """
    note this only checks the first param and from that infers the rest is in gpu.

    https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180
    :return:
    """
    device = next(mdl.parameters()).device
    return device


def get_device_from(mdl) -> torch.device:
    """
    Checks the device of the first set of params.

    https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180
    :return:
    """
    device: torch.device = next(mdl.parameters()).device
    return device


def get_device(gpu_idx: int = 0) -> torch.device:
    """
    Get default gpu torch device.

    :param gpu_idx:
    :return:
    """
    device: torch.device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    # device: torch.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
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
    """
    create a deep detached copy of mdl_new.
    Needs the human_mdl (instantiated by a human) as an empty vessel and then
    copy the parameters from the real model we want (mdl_to_copy) and returns a filled in
    copy of the human_mdl.
    Essentially does:
    empty_vessel_mdl = deep_copy(human_mdl)
    mdl_new.fill() <- copy(from=mdl_to_copy,to=human_mdl)
    """
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
        # w_new = nn.Parameter(w_new)
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
        bidirectional: (torch_uu.Tensor) initial hidden state (n_layers*nb_directions, batch_size, hidden_size)

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
    :return torch_uu.Tensor hidden: initial hidden state (n_layers*nb_directions, batch_size, hidden_size)
    """
    # get gpu
    use_cuda = torch.cuda.is_available()
    device_gpu_if_avail = torch.device("cuda" if use_cuda else "cpu")
    device = device if device == None else device_gpu_if_avail
    ## get initial memory and hidden cell (c and h)
    nb_directions = 2 if bidirectional else 1
    h_n = torch.randn(nb_layers * nb_directions, batch_size, hidden_size, device=device)
    c_n = torch.randn(nb_layers * nb_directions, batch_size, hidden_size, device=device)
    hidden = (h_n, c_n)
    return hidden


def lp_norm(mdl: nn.Module, p: int = 2) -> Tensor:
    lp_norms = [w.norm(p) for name, w in mdl.named_parameters()]
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
    for name, w in mdl.named_parameters():
        all_leafs = all_leafs and w.is_leaf
    return all_leafs


def calc_error(mdl: torch.nn.Module, X: torch.Tensor, Y):
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
    max_scores, max_idx_class = y_logits.max(
        dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
    # usually 0th coordinate is batch size
    n = X.size(0)
    assert (n == max_idx_class.size(0))
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
    assert (n_examples == max_indices_classes.size(0))
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
        correct = (
                y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(
                -1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0,
                                                                                        keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        if len(list_topk_accs) == 1:
            return list_topk_accs[0]  # only the top accuracy you requested
        else:
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
        correct = (
                y_pred == target_reshaped)  # [B, maxk] were for each example we know which topk prediction matched truth

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:, :k]  # [B, maxk] -> [B, maxk]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(
                -1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0,
                                                                                        keepdim=True)  # [kB] -> [1]
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
        flatten_tensor {torchTensor} -- torch_uu tensor to get stats
    
    Returns:
        [list torch_uu.Tensor] -- [mu, std, min_v, max_v, med]
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

# def save_ckpt_meta_learning(args, meta_learner, debug=False):
#     # https://discuss.pytorch.org/t/advantages-disadvantages-of-using-pickle-module-to-save-models-vs-torch-save/79016
#     # make dir to logs (and ckpts) if not present. Throw no exceptions if it already exists
#     path_to_ckpt = args.logger.current_logs_path
#     path_to_ckpt.mkdir(parents=True, exist_ok=True) # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.
#     ckpt_path_plus_path = path_to_ckpt / Path('db')
#
#     # Pickle args & logger (note logger is inside args already), source: https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
#     db = {} # database dict
#     tb = args.tb
#     args.tb = None
#     args.base_model = "no child mdl in args see meta_learner" # so that we don't save the child model so many times since it's part of the meta-learner
#     db['args'] = args # note this obj has the last episode/outer_i we ran
#     db['meta_learner'] = meta_learner
#     torch.save(db, ckpt_path_plus_path)
#     # with open(ckpt_path_plus_path , 'wb+') as db_file:
#     #     pickle.dump(db, db_file)
#     if debug:
#         test_ckpt_meta_learning(args, meta_learner, debug)
#     args.base_model = meta_learner.base_model # need to re-set it otherwise later in the code the pointer to child model will be updated and code won't work
#     args.tb = tb
#     return

def load(path: Union[Path, str], filename: str, pickle_module=dill):
    if isinstance(path, str):
        path = Path(path).expanduser()
    else:
        path = path.expanduser()
    data = torch.load(path / filename, pickle_module=pickle_module)
    return data


def save_checkpoint_simple(args, meta_learner):
    # make dir to logs (and ckpts) if not present. Throw no exceptions if it already exists
    args.path_2_save_ckpt.mkdir(parents=True,
                                exist_ok=True)  # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.

    args.base_model = "check the meta_learner field in the checkpoint not in the args field"  # so that we don't save the child model so many times since it's part of the meta-learner
    # note this obj has the last episode/outer_i we ran
    torch.save({'args': args, 'meta_learner': meta_learner}, args.path_2_save_ckpt / Path('/ckpt_file'))


def _save(agent, epoch_num: int, dirname=None, ckpt_filename: str = 'mdl.pt'):
    """
    Saves checkpoint for any worker.
    Intended use is to save by worker that got a val loss that improved.

    todo - goot save ckpt always with epoch and it, much sure epoch_num and it are
    what you intended them to be...
    """
    from uutils.torch_uu.distributed import is_lead_worker
    if is_lead_worker(agent.args.rank):
        import uutils
        from torch.nn.parallel.distributed import DistributedDataParallel
        dirname = agent.args.log_root if dirname is None else dirname
        pickable_args = uutils.make_args_pickable(agent.args)
        import dill
        mdl = agent.mdl.module if type(agent.mdl) is DistributedDataParallel else agent.mdl
        # self.remove_term_encoder_cache()
        torch.save({'state_dict': agent.mdl.state_dict(),
                    'epoch_num': epoch_num,
                    'it': agent.args.it,
                    'optimizer': agent.optimizer.state_dict(),
                    'args': pickable_args,
                    'mdl': mdl},
                   pickle_module=dill,
                   f=dirname / ckpt_filename)  # f'mdl_{epoch_num:03}.pt'


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


MetaLearner = object


def get_model_opt_meta_learner_to_resume_checkpoint_resnets_rfs(args: Namespace,
                                                                path2ckpt: str,
                                                                filename: str,
                                                                device: Optional[torch.device] = None,
                                                                # precedence_to_args_checkpoint: bool = True,
                                                                ) -> tuple[nn.Module, optim.Optimizer, MetaLearner]:
    """
    Get the model, optimizer, meta_learner to resume training from checkpoint.

    Examples:
        - see: _resume_from_checkpoint_meta_learning_for_resnets_rfs_test

    ref:
        - https://stackoverflow.com/questions/70129895/why-is-it-not-recommended-to-save-the-optimizer-model-etc-as-pickable-dillable
    """
    import uutils
    path2ckpt: Path = Path(path2ckpt).expanduser() if isinstance(path2ckpt, str) else path2ckpt.expanduser()
    ckpt: dict = torch.load(path2ckpt / filename, map_location=torch.device('cpu'))
    # - args
    # args_ckpt: Namespace = ckpt['args']
    # if args_ckpt is not None:
    #     if precedence_to_args_checkpoint:
    #         args: Namespace = uutils.merge_args(starting_args=args, updater_args=args_ckpt)
    #     else:
    #         args: Namespace = uutils.merge_args(starting_args=args_ckpt, updater_args=args)
    # -
    training_mode = ckpt.get('training_mode')
    if training_mode is not None:
        assert uutils.xor(training_mode == 'epochs', training_mode == 'iterations')
        if training_mode == 'epochs':
            args.epoch_num = ckpt['epoch_num']
        else:
            args.it = ckpt['it']
    # - get meta-learner
    meta_learner: MetaLearner = ckpt['meta_learner']
    # - get model
    model: nn.Module = meta_learner.base_model
    # - get outer-opt
    outer_opt_str = ckpt.get('outer_opt_str')
    if outer_opt_str is not None:
        # use the string to create optimizer, load the state dict, etc.
        outer_opt: optim.Optimizer = get_optimizer(outer_opt_str)
        outer_opt_state_dict: dict = ckpt['outer_opt_state_dict']
        outer_opt.load_state_dict(outer_opt_state_dict)
    else:
        # this is not ideal, but since Adam has a exponentially moving average for it's adaptive learning rate,
        # hopefully this doesn't screw my checkpoint to much
        outer_opt: optim.Optimizer = optim.Adam(model.parameters(), lr=args.outer_lr)
    # - device setup
    if device is not None:
        # if torch.cuda.is_available():
        #     meta_learner.base_model = meta_learner.base_model.cuda()
        meta_learner.base_model.to(device)
        meta_learner.to(device)
    return model, outer_opt, meta_learner


def get_optimizer(optimizer_name: str) -> optim.Optimizer:
    raise ValueError('Not implemented')


def _load_model_and_optimizer_from_checkpoint(args: Namespace, training: bool = True) -> Namespace:
    """
    based from: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

    ref:
        - https://stackoverflow.com/questions/70129895/why-is-it-not-recommended-to-save-the-optimizer-model-etc-as-pickable-dillable
    """
    import torch
    from torch import optim
    import torch.nn as nn
    # model = Net()
    args.model = nn.Linear()
    # optimizer = optim.SGD(args.model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(args.model.parameters(), lr=0.001)

    # scheduler...

    checkpoint = torch.load(args.PATH)
    args.model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    args.epoch_num = checkpoint['epoch_num']
    args.loss = checkpoint['loss']

    args.model.train() if training else args.model.eval()


def ckpt_meta_learning_test(args, meta_learner, verbose=False):
    path_to_ckpt = args.logger.current_logs_path
    path_to_ckpt.mkdir(parents=True,
                       exist_ok=True)  # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.
    ckpt_path_plus_path = path_to_ckpt / Path('db')

    # Pickle args & logger (note logger is inside args already), source: https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
    db = {}  # database dict
    args.base_model = "no child mdl"  # so that we don't save the child model so many times since it's part of the meta-learner
    db['args'] = args  # note this obj has the last episode/outer_i we ran
    args.base_model = meta_learner.base_model  # need to re-set it otherwise later in the code the pointer to child model will be updated and code won't work
    db['meta_learner'] = meta_learner
    with open(ckpt_path_plus_path, 'wb+') as db_file:
        dumped_outer_i = args.outer_i
        pickle.dump(db, db_file)
    with open(ckpt_path_plus_path, 'rb') as db_file:
        args = get_args_debug(path=path_to_ckpt)
        loaded_outer_i = args.outer_i
    if verbose:
        print(f'==> dumped_outer_i = {dumped_outer_i}')
        print(f'==> loaded_outer_i = {loaded_outer_i}')
    ## Assertion Tests
    assert (dumped_outer_i == loaded_outer_i)
    return


def ckpt_test(args, verbose=False):
    path_to_ckpt = args.logger.current_logs_path
    path_to_ckpt.mkdir(parents=True,
                       exist_ok=True)  # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.
    ckpt_path_plus_path = path_to_ckpt / Path('db')

    ## Pickle args & logger (note logger is inside args already), source: https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
    db = {}  # database dict
    db['args'] = args  # note this obj has the last episode/outer_i we ran
    with open(ckpt_path_plus_path, 'wb+') as db_file:
        dumped_outer_i = args.outer_i
        pickle.dump(db, db_file)
    with open(ckpt_path_plus_path, 'rb') as db_file:
        args = get_args_debug(path=path_to_ckpt)
        loaded_outer_i = args.outer_i
    if verbose:
        print(f'==> dumped_outer_i = {dumped_outer_i}')
        print(f'==> loaded_outer_i = {loaded_outer_i}')
    ## Assertion Tests
    assert (dumped_outer_i == loaded_outer_i)
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


def set_system_wide_force_flush2():
    """
    Force flushes the entire print function everywhere.

    https://stackoverflow.com/questions/230751/how-to-flush-output-of-print-function
    :return:
    """
    import builtins
    import functools
    print2 = functools.partial(print, flush=True)
    builtins.print = print2


def resume_ckpt_meta_lstm(metalearner, optim, resume, device):
    ckpt = torch.load(resume, map_location=device)
    last_episode = ckpt['episode']
    metalearner.load_state_dict(ckpt['metalearner'])
    optim.load_state_dict(ckpt['optim'])
    return last_episode, metalearner, optim


def train_single_batch_agent(agent, train_batch, val_batch, acc_tolerance=1.0, train_loss_tolerance=0.01):
    """
    Train untils the accuracy on the specified batch has perfect interpolation in loss and accuracy.
    It also prints and tb logs every iteration.


    todo - compare with train one batch

    :param acc_tolerance:
    :param train_loss_tolerance:
    :return:
    """
    import progressbar
    import uutils

    set_system_wide_force_flush2()

    # train_batch = next(iter(agent.dataloaders['train']))
    # val_batch = next(iter(agent.dataloaders['val']))

    def log_train_stats(it: int, train_loss: float, acc: float):
        val_loss, val_acc = agent.forward_one_batch(val_batch, training=False)
        agent.log_tb(it=it, tag1='train loss', loss=float(train_loss), tag2='train acc', acc=float(acc))
        agent.log_tb(it=it, tag1='val loss', loss=float(val_loss), tag2='val acc', acc=float(val_acc))

        agent.log(f"\n{it=}: {train_loss=} {acc=}")
        agent.log(f"{it=}: {val_loss=} {val_acc=}")

    # first batch
    avg_loss = AverageMeter('train loss')
    avg_acc = AverageMeter('train accuracy')
    agent.args.it = 0
    bar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    while True:
        train_loss, train_acc = agent.forward_one_batch(train_batch, training=True)

        agent.optimizer.zero_grad()
        train_loss.backward()  # each process synchronizes it's gradients in the backward pass
        agent.optimizer.step()  # the right update is done since all procs have the right synced grads

        # if agent.agent.is_lead_worker() and agent.args.it % 10 == 0:
        if agent.args.it % 10 == 0:
            bar.update(agent.args.it)
            log_train_stats(agent.args.it, train_loss, train_acc)
            agent.save(
                agent.args.it)  # very expensive! since your only fitting one batch its ok to save it every time you log - but you might want to do this left often.

        agent.args.it += 1
        gc.collect()
        # if train_acc >= acc_tolerance and train_loss <= train_loss_tolerance:
        if train_acc >= acc_tolerance:
            log_train_stats(agent.args.it, train_loss, train_acc)
            agent.save(
                agent.args.it)  # very expensive! since your only fitting one batch its ok to save it every time you log - but you might want to do this left often.
            bar.update(agent.args.it)
            break  # halt once performance is good enough

    return avg_loss.item(), avg_acc.item()


def train_single_batch(args, agent, mdl, optimizer, acc_tolerance=1.0, train_loss_tolerance=0.01):
    """
    Train untils the accuracy on the specified batch has perfect interpolation in loss and accuracy.
    It also prints and tb logs every iteration.

    :param acc_tolerance:
    :param train_loss_tolerance:
    :return:
    """
    from uutils.torch_uu.distributed import process_batch_ddp_tactic_prediction

    print('train_single_batch')
    set_system_wide_force_flush2()
    avg_loss = AverageMeter('train loss')
    avg_acc = AverageMeter('train accuracy')

    def forward_one_batch(data_batch, training):
        mdl.train() if training else mdl.eval()
        data_batch = process_batch_ddp_tactic_prediction(args, data_batch)
        loss, logits = mdl(data_batch)
        acc = accuracy(output=logits, target=data_batch['tac_label'])
        avg_loss.update(loss.item(), args.batch_size)
        avg_acc.update(acc.item(), args.batch_size)
        return loss, acc

    def log_train_stats(it: int, train_loss: float, acc: float):
        val_loss, val_acc = forward_one_batch(val_batch, training=False)
        # agent.log_tb(it=it, tag1='train loss', loss=float(train_loss), tag2='train acc', acc=float(acc))
        # agent.log_tb(it=it, tag1='val loss', loss=float(val_loss), tag2='val acc', acc=float(val_acc))

        print(f"\n{it=}: {train_loss=} {acc=}")
        print(f"{it=}: {val_loss=} {val_acc=}")

    # train_acc = 0.0; train_loss = float('inf')
    data_batch = next(iter(agent.dataloaders['train']))
    val_batch = next(iter(agent.dataloaders['val']))
    args.it = 0
    while True:
        train_loss, train_acc = mdl.forward_one_batch(data_batch, training=True)

        optimizer.zero_grad()
        train_loss.backward()  # each process synchronizes it's gradients in the backward pass
        optimizer.step()  # the right update is done since all procs have the right synced grads

        # if args.it % 10 == 0:
        #     log_train_stats(args.it, train_loss, train_acc)
        #     agent.save(args.it)  # very expensive! since your only fitting one batch its ok to save it every time you log - but you might want to do this left often.

        args.it += 1
        gc.collect()
        # if train_acc >= acc_tolerance and train_loss <= train_loss_tolerance:
        if train_acc >= acc_tolerance:
            log_train_stats(args.it, train_loss, train_acc)
            # agent.save(args.it)  # very expensive! since your only fitting one batch its ok to save it every time you log - but you might want to do this left often.
            break  # halt once both the accuracy is high enough AND train loss is low enough

    return avg_loss.item(), avg_acc.item()


##


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
    # do gradient clipping: If ‖g‖ ≥ c Then g := c * g/‖g‖
    # note: grad_clip_rate is a number for clipping the other is the type
    # of clipping we are doing
    if hasattr(args, 'grad_clip_rate'):
        if args.grad_clip_rate is not None:
            if args.grad_clip_mode == 'clip_all_seperately':
                for group_idx, group in enumerate(meta_opt.param_groups):
                    for p_idx, p in enumerate(group['params']):
                        nn.utils.clip_grad_norm_(p, args.grad_clip_rate)
            elif args.grad_clip_mode == 'clip_all_together':
                # [y for x in list_of_lists for y in x]
                all_params = [p for group in meta_opt.param_groups for p in group['params']]
                nn.utils.clip_grad_norm_(all_params, args.grad_clip_rate)
            elif args.grad_clip_mode == 'no_grad_clip' or args.grad_clip_mode is None:  # i.e. do not clip if grad_clip_rate is None
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
        x {[torch_uu.Tensor]} -- input to preprocess
    
    Keyword Arguments:
        p {int} -- number that indicates the scaling (default: {10})
        eps {float} - numerical stability param (default: {1e-8})
    
    Returns:
        [torch_uu.Tensor] -- preprocessed numbers
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


# - distances

def functional_diff_norm(f1, f2, lb=-1.0, ub=1.0, p=2):
    """
    Computes norm:

    ||f||_p = (int_S |f|^p dmu)^1/p

    https://en.wikipedia.org/wiki/Lp_space

    https://stackoverflow.com/questions/63237199/how-does-one-compute-the-norm-of-a-function-in-python
    """
    # index is there since it also returns acc/err
    if 'torch_uu' in str(type(f1)) or 'torch_uu' in str(type(f2)):
        pointwise_diff = lambda x: abs(f1(torch.tensor([x])) - f2(torch.tensor([x]))) ** p
    else:
        pointwise_diff = lambda x: abs(f1(x) - f2(x)) ** p
    norm, abs_err = integrate.quad(pointwise_diff, a=lb, b=ub)
    return norm ** (1 / p), abs_err


def get_metric(mdl1: nn.Module, mdl2: nn.Module,
               X1: Tensor, X2: Tensor,
               layer_name: str,
               metric_comparison_type: str = 'pwcca',
               iters: int = 1,
               effective_neuron_type: str = 'filter',
               downsample_method: Optional[str] = None,
               downsample_size: Optional[int] = None,
               subsample_effective_num_data_method: Optional[str] = None,
               subsample_effective_num_data_param: Optional[int] = None,
               metric_as_sim_or_dist: str = 'dist'
               ) -> float:
    """
    Computes distance between layer matrices for a specific layer:
        d: float = dist(mdl1(X1), mdl2(X2))  # d = 1 - sim

    :argument: cxa_dist_type 'svcca', 'pwcca', 'lincka', 'opd'.
    """
    from anatome import SimilarityHook as DistanceHook
    # - get distance hooks (to intercept the features)
    hook1 = DistanceHook(mdl1, layer_name, metric_comparison_type)
    hook2 = DistanceHook(mdl2, layer_name, metric_comparison_type)
    mdl1.eval()
    mdl2.eval()
    # - populate hook tensors in the data dimension (1st dimension):
    # so it populates the self._hooked_tensors in the hook objects.
    for _ in range(iters):  # might make sense to go through multiple is NN is stochastic e.g. BN, dropout layers
        mdl1(X1)
        mdl2(X2)
    # - compute distiance with hooks
    dist: float = hook1.distance(hook2,
                                 effective_neuron_type=effective_neuron_type,
                                 downsample_method=downsample_method,
                                 downsample_size=downsample_size,
                                 subsample_effective_num_data_method=subsample_effective_num_data_method,
                                 subsample_effective_num_data_param=subsample_effective_num_data_param,
                                 metric_as_sim_or_dist=metric_as_sim_or_dist
                                 )
    # - remove hook, to make sure code stops being stateful (I hope)
    hook1.clear()
    hook2.clear()
    remove_hook(mdl1, hook1)
    remove_hook(mdl2, hook2)
    return float(dist)


def ned(f, y):
    """
    Normalized euncleadian distance

    ned = sqrt 0.5*np.var(x - y) / (np.var(x) + np.var(y)) = 0.5 variance of difference / total variance individually

    reference: https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance

    @param x:
    @param y:
    @return:
    """
    ned = (0.5 * np.var(f - y) / (np.var(f) + np.var(y))) ** 0.5
    return ned


def r2_score_from_torch(y_true: torch.Tensor, y_pred: torch.Tensor):
    """ returns the accuracy from torch_uu tensors """
    from sklearn.metrics import r2_score
    acc = r2_score(y_true=y_true.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy())
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
        r2 = 0.5 * r2_f + 0.5 * r2_y
    elif r2_type == 'normalized_average_r2s':
        r2_f = r2_score(y_true=f, y_pred=y)
        r2_y = r2_score(y_true=y, y_pred=f)
        r2 = 0.5 * r2_f + 0.5 * r2_y
        # sig = torch_uu.nn.Sigmoid()
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


def compressed_r2_score(y_true, y_pred, compressor='tanh'):
    """
    The idea is that the negative part of the r2_score doesn't go to -infinity.
    Once it is negative we know we it's performing at chance and the predictor is trash.
    So for the sake of making things easier to plot we squish everything bellow r2<0 with a tanh
    by default so r2 is bounded between [-1,1] where 0 is chance
    (as good as predicting average target y without using any features from x).
    Interpretation
        - cr2>0 standard r2 interpretation
        - cr2=0 standard horizontal chance interpretation
        - cr2 < 0 squished r2 score (all negative values are very bad)

    compressed_r2_score(r2) =
    {
        r2   if r2 >0,
        0   if r2==0,
        tanh(r2)   if r2 <0
    }

    If compressor=Sigmoid then
        - cr2>0.5 standard r2 interpretation
        - cr2=0.5 standard horizontal chance interpretation
        - cr2 < 0.5 squished r2 score (all values bellow 0.5 are very bad)

    compressed_r2_score(r2, Sigmoid) =
    {
        0.5r2+0.5 if r2 >0,
        0 if r2==0,
        tanh(r2) if r2 <0
    }

    :param compressor: Sigmoid otherwise use Tanh
    :param y_true:
    :param y_pred:
    :return:
    """
    from sklearn.metrics import r2_score
    from scipy.stats import logistic

    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    if compressor == 'Sigmoid':
        if r2 > 0:
            # so that cr2 intercepts at 0.5
            compressed_r2 = 0.5 * r2 + 0.5
        else:
            compressed_r2 = logistic.cdf(r2)
    elif compressor == 'tanh':
        if r2 > 0:
            compressed_r2 = r2
        else:
            compressed_r2 = np.tanh(r2)
    else:
        raise ValueError(f'compressor {compressor} not implemented')
    return compressed_r2


def compressed_r2_score_from_torch(y_true: torch.Tensor, y_pred: torch.Tensor, compressor='tanh'):
    """
    Though it seems this function is not needed, surprisingly! It processes torch_uu tensors just fine...
    :param y_true:
    :param y_pred:
    :param compressor:
    :return:
    """
    return compressed_r2_score(y_true.detach().numpy(), y_pred.detach().numpy(), compressor)


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
#     sig = torch_uu.nn.Sigmoid() if normalizer == 'Sigmoid' else torch_uu.nn.Tanh()
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
#         x = torch_uu.torch_uu.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
#         mdl1(x)
#         mdl2(x)
#     dist = hook1.distance(hook2, size=cca_size)
#     return dist

# def cca(mdl1, mdl2, dataloader, cca_size=None, iters=10):
#     # with torch_uu.no_grad()
#     for _ in range(iters):
#         next()
#         mdl1(x)
#         mdl2(x)

def l2_sim_torch(x1, x2, dim=1, sim_type='nes_torch') -> Tensor:
    if sim_type == 'nes_torch':
        sim = nes_torch(x1, x2, dim)
    elif sim_type == 'cosine_torch':
        cos = nn.CosineSimilarity(dim=dim)
        sim = cos(x1, x2)
    elif sim_type == 'op_torch':
        sim = orthogonal_procrustes_similairty(x1, x2, normalize_for_range_0_to_1=True)
    else:
        raise ValueError(f'Not implemented sim_type={sim_type}')
    return sim


def ned_torch(x1: torch.Tensor, x2: torch.Tensor, dim=0, eps=1e-8) -> Tensor:
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
    assert False, f'Need to test if dim=0 is correct...'
    # to compute ned for two individual vectors e.g to compute a loss (NOT BATCHES/COLLECTIONS of vectorsc)
    if len(x1.size()) == 1:
        # [K] -> [1]
        ned_2 = 0.5 * ((x1 - x2).var() / (x1.var() + x2.var() + eps))
    # if the input is a (row) vector e.g. when comparing two batches of acts of D=1 like with scores right before sf
    elif x1.size() == torch.Size(
            [x1.size(0), 1]):  # note this special case is needed since var over dim=1 is nan (1 value has no variance).
        # [B, 1] -> [B]
        ned_2 = 0.5 * ((x1 - x2) ** 2 / (
                x1 ** 2 + x2 ** 2 + eps)).squeeze()  # Squeeze important to be consistent with .var, otherwise tensors of different sizes come out without the user expecting it
    # common case is if input is a batch
    else:
        # e.g. [B, D] -> [B]
        ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5


def nes_torch(x1, x2, dim: int = 0, eps: float = 1e-8) -> Tensor:
    return 1.0 - ned_torch(x1, x2, dim, eps)


def orthogonal_procrustes_distance(x1: Tensor, x2: Tensor, normalize_for_range_0_to_1: bool = True) -> Tensor:
    """
    Computes the orthoginal procrustes distance.
    If normalized then the answer is divided by 2 so that it's in the interval [0, 1].
    Outputs a single number with no dimensionality.

    Expected input:
        - two matrices e.g.
            - two weight matrices of size [num_weights1, num_weights2]
            - or two matrices of activations [batch_size, dim_of_layer] (used by paper [1])

    d_proc(A*, B) = ||A||^2_F + ||B||^2_F - 2||A^T B||_*
    || . ||_* = nuclear norm = sum of singular values sum_i sig(A)_i = ||A||_*

    Note:
    - this only works for matrices. So it's works as a metric for FC and transformers (or at least previous work
    only used it for transformer [1] which have FC and no convolutions.
    - note Ding et. al. say: for a raw representation A we first subtract the mean value from each column, then divide
    by the Frobenius norm, to produce the normalized representation A* , used in all our dissimilarity computation.
        - which is different from dividing by the variance, which is what I would have expected.

    ref:
    - [1] https://arxiv.org/abs/2108.01661
    - [2] https://discuss.pytorch.org/t/is-there-an-orthogonal-procrustes-for-pytorch/131365
    - [3] https://ee227c.github.io/code/lecture5.html#nuclear-norm

    sample output (see test) - so it outputs a number inside a tensor obj:
    [('fc0', tensor(5.7326, grad_fn=<RsubBackward1>)),
     ('ReLU0', tensor(2.6101, grad_fn=<RsubBackward1>)),
     ('fc1', tensor(3.8898, grad_fn=<RsubBackward1>)),
     ('ReLU2', tensor(1.3644, grad_fn=<RsubBackward1>)),
     ('fc3', tensor(1.5007, grad_fn=<RsubBackward1>))]

    :param x1:
    :param x2:
    :return:
    """
    from torch.linalg import norm
    x1, x2 = normalize_matrix_for_similarity(x1, dim=1), normalize_matrix_for_similarity(x2, dim=1)
    x1x2 = x1.t() @ x2
    d: Tensor = norm(x1, 'fro') + norm(x2, 'fro') - 2 * norm(x1x2, 'nuc')
    d: Tensor = d / 2.0 if normalize_for_range_0_to_1 else d
    return d


def orthogonal_procrustes_similairty(x1: Tensor, x2: Tensor, normalize_for_range_0_to_1: bool = True) -> Tensor:
    """
    Returns orthogonal procurstes similarity. If normalized then output is in invertval [0, 1] and if not then output
    is in interval [0, 1]. See orthogonal_procrustes_distance for details and references.

    :param x1:
    :param x2:
    :param normalize:
    :return:
    """
    d = orthogonal_procrustes_distance(x1, x2, normalize_for_range_0_to_1)
    sim: Tensor = 1.0 - d if normalize_for_range_0_to_1 else 2.0 - d
    return sim


def normalize_matrix_for_similarity(X: Tensor, dim: int = 0) -> Tensor:
    """
    Normalize matrix of size wrt to the data dimension according to Ding et. al.
        X_normalized = X_centered / ||X_centered||_F
    Assumption is that X is of size [n, d].
    Otherwise, specify which dimension to normalize with dim.
    This gives more accurate results for OPD when normalizing by the centered data.

    Note:
        - this does not normalize by the std of the data.
        - not centering produces less accurate results for OPD.
    ref: https://stats.stackexchange.com/questions/544812/how-should-one-normalize-activations-of-batches-before-passing-them-through-a-si
    """
    from torch.linalg import norm
    # X_centered: Tensor = (X - X.mean(dim=dim, keepdim=True))
    X_centered: Tensor = center(X, dim)
    X_star: Tensor = X_centered / norm(X_centered, "fro")
    return X_star


def center(input: Tensor,
           dim: int
           ) -> Tensor:
    return _zero_mean(input, dim)


def _zero_mean(input: Tensor,
               dim: int
               ) -> Tensor:
    return input - input.mean(dim=dim, keepdim=True)


def normalize_matrix_for_distance(X: Tensor, dim: int = 0) -> Tensor:
    """ Center according to columns and divide by frobenius norm. Matrix is assumed to be [n, d] else sepcify dim. """
    return normalize_matrix_for_similarity(X, dim)


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
    # recursive case, for every sub list get it into tensor (recursively) form and then combine with torch_uu.stack
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


def print_results_old(args, all_meta_eval_losses, all_diffs_qry, all_diffs_cca, all_diffs_cka, all_diffs_neds):
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


def print_results(args, all_meta_eval_losses, all_diffs_qry, all_diffs_cca, all_diffs_cka, all_diffs_neds):
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
    stats = {metric: {'avg': None, 'std': None, 'rep': {'avg': None, 'std': None}, 'all': {'avg': None, 'std': None}}
             for metric, _ in all_sims.items()}
    for metric, tensor_of_metrics in all_sims.items():
        if metric in cxas:
            # compute average cxa per layer: [T, L] -> [L]
            avg_sims = tensor_of_metrics.mean(dim=0)
            std_sims = tensor_of_metrics.std(dim=0)
            # compute representation & all avg cxa [T, L] -> [1]
            L = tensor_of_metrics.size(1)
            indicies = torch.tensor(range(L - 1))
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
            indicies = torch.tensor(range(L - 1))
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
        if metric in cxas + l2:
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

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        """
        Note: usually n=batch_size so that we can keep track of the total sum.
        If you don't log the batch size the quantity your tracking is the average of the sample means
        which has the same expectation but because your tracking emperical estimates you will have a
        different variance. Thus, it's recommended to have n=batch_size

        :param val:
        :param n: usually the batch size
        :return:
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def item(self):
        if type(self.avg) is torch.Tensor:
            return self.avg.item()
        else:
            return float(self.avg)

    def __str__(self):
        fmtstr = '{name} val:{val} avg:{avg}'
        return fmtstr.format(**self.__dict__)


class StatsCollector(object):
    """Computes stores the average and the std. """

    def __init__(self):
        pass

    def __init__(self):
        pass

    def append(self, val: int):
        pass

    def compute_stats(self):
        """Retrun avg and std """
        pass

    def __str__(self):
        pass
        # fmtstr = f'{name} val:{val} avg:{avg}'
        # return fmtstr.format(**self.__dict__)


def flatten2float_list(t: torch.Tensor) -> List[float]:
    """
    Maps a tensor to a flatten list of floats
    :param t:
    :return:
    """
    t = t.view(-1).detach().numpy().tolist()
    # t = torch_uu.flatten(t).numpy().tolist()
    return t


# -- not using for now

class AverageStdMeter(object):
    """Computes and stores the average & std and current value.

    I decided not to use this because it might make ppl do mistakes. For example if you are
    logging here the val as the batch loss then this class would compute the variance of the
    sample mean. This will be an under estimate of the quantity you (usually) really care about which
    is the variance of the loss on new examples (not the average). The reason you want this is as follows:
    in ML usually we really want is to be able to predict well unseen examples. So we really want a low
    l(f(x), y) for a new example. So the RV we are interested is l for a single example. In meta-learning
    it would be similar. We want to have a low loss on a new task. If this is the quantity we care about
    then computing the spread over a set of batch means would underestimate the quantity we really care about.
    Thus, unless you can get a flatten list of losses for each example (which libraries like pytorch don't
    usually give) this object might not be as useful as one might think.
    Note for meta-learning you'd get very bad estimates of the std since this class would return the std of
    the sample mean std_N = st/sqrt{N} which is an underestimate by sqrt{N}. You might be expecting to
    measure the std (the spread of the losses of tasks) but you wouldn't actually be getting that.
    You could multiply it by sqrt{N} but that introduce more errors since your std_N is usually computed
    using a sample (not the true Var[] operator). You might be tempted to use N=1 and that would work
    if your using the true Var[] opertor but if your computing it emprically you'd get NaN since the
    variance is undefined for 1 element.
    """

    def __init__(self, name):
        self.name = name
        raise ValueError('Dont use.')

    def reset(self):
        self.vals = []
        self.avg = 0
        self.std = 0

    def update(self, val):
        self.vals.append(val)
        self.avg = np.mean(self.vals)
        self.std = np.std(self.vals)

    def items(self):
        return self.avg, self.std

    def __str__(self):
        fmtstr = '{name} {avg} +- {std}'
        return fmtstr.format(self.name, self.avg, self.std)


def split_train_val_test(X, y, random_state=1, ratio=[0.80, 0.10, 0.10]):
    #
    # # shuffle = False  # shufflebool, default=True, Whether or not to shuffle the data_lib before splitting. If shuffle=False then stratify must be None.
    # X_train, X_val_test, y_train, y_val_test = train_test_split(X, y,
    #                                                             test_size=test_size,
    #                                                             random_state=random_state)
    # print(len(X_train))
    # print(len(X_val_test))
    #
    # # then 2/3 for val, 1/3 for test to get 10:5 split
    # test_size = 1.0 / 3.0
    # X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_size,
    #                                                  random_state=random_state)
    # return X_train, X_val, X_test, y_train, y_val, y_test
    pass


def split_two(lst, ratio=[0.5, 0.5]):
    assert (np.sum(ratio) == 1.0)  # makes sure the splits make sense
    train_ratio = ratio[0]
    # note this function needs only the "middle" index to split, the remaining is the rest of the split
    indices_for_splittin = [int(len(lst) * train_ratio)]
    train, test = np.split(lst, indices_for_splittin)
    return train, test


def split_three(lst, ratio=[0.8, 0.1, 0.1]):
    import numpy as np

    train_r, val_r, test_r = ratio
    assert (np.sum(ratio) == 1.0)  # makes sure the splits make sense
    # note we only need to give the first 2 indices to split, the last one it returns the rest of the list or empty
    indicies_for_splitting = [int(len(lst) * train_r), int(len(lst) * (train_r + val_r))]
    train, val, test = np.split(lst, indicies_for_splitting)
    return train, val, test


# -- Label smoothing

"""
refs:
https://discuss.pytorch.org/t/labels-smoothing-and-categorical-loss-functions-alternatives/11339/12
https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580
https://github.com/pytorch/pytorch/issues/7455
https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186
https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch

"""


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


def train_one_batch(opts, model, train_batch, val_batch, optimizer, tolerance=0.01):
    """
    Code for training a generic pytorch model with one abtch.
    The idea is that the user uses (their perhaps custom) data loader to sample a data batch once
    and then pass them to this code to train until the batch has been overfitted.

    Note: If you are doing regression you will have to adapt this code - however, I do recommend that
    you track some sort of accuracy for your regression task. For example, track R2 (or some squeezed
    version of it). This is really useful because your loss has an arbitrary uninterpretable scale
    while R2 always has a nice interpretation (how far you are from just predicting the mean target y
    of your data without using any features x). Having this sort of interpretable measure can save you
    a lot of time - especially when the loss seems to be meaninfuless.
    For that replace accuracy for your favorite interpretable "acuraccy" function.


    todo - compare with train_single_batch

    :param opts:
    :param model:
    :param train_batch:
    :param val_batch:
    :param optimizer:
    :param tolerance:
    :return:
    """
    avg_loss = AverageMeter('train loss')
    avg_acc = AverageMeter('train accuracy')

    it = 0
    train_loss = float('inf')
    x_train_batch, y_train_batch = train_batch
    x_val_batch, y_val_batch = val_batch
    while train_loss > tolerance:
        model.train()
        train_loss, logits = model(x_train_batch)
        avg_loss.update(train_loss.item(), opts.batch_size)
        train_acc = accuracy(output=logits, target=y_train_batch)
        avg_acc.update(train_acc.item(), opts.batch_size)

        optimizer.zero_grad()
        train_loss.backward()  # each process synchronizes it's gradients in the backward pass
        optimizer.step()  # the right update is done since all procs have the right synced grads

        model.eval()
        val_loss, logits = model(x_train_batch)
        avg_loss.update(train_loss.item(), opts.batch_size)
        val_acc = accuracy(output=logits, target=y_val_batch)
        avg_acc.update(val_acc.item(), opts.batch_size)

        log_2_tb(it=it, tag1='train loss', loss=train_loss.item(), tag2='train acc', acc=train_acc.item())
        log_2_tb(it=it, tag1='val loss', loss=val_loss.item(), tag2='val acc', acc=val_acc.item())
        print(f"\n{it=}: {train_loss=} {train_acc=}")
        print(f"{it=}: {val_loss=} {val_acc=}")

        it += 1
        gc.collect()

    return avg_loss.item(), avg_acc.item()


# - data set downloads

def download_dataset(url: str, path2save_filename: Union[str, None] = None,
                     do_unzip: bool = False) -> None:
    """

    :param url:
    :param path2save: the path to save and the filename of the file in one string e.g. ~/data.zip.
    :return:
    """
    if path2save_filename is None:
        filename: str = url.split('/')[-1]
        filename = Path(f'./{filename}').expanduser()
        # todo , this doesnt actually log the right path
        #  get file path of where code is executing and where data set will be saved
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # logging.warning(f'Your data set will be saved in the directory {dir_path}.')
    else:
        filename: str = Path(path2save_filename).expanduser()
    print(f'data set downloaded to path with filename: {path2save_filename=}')
    urllib.request.urlretrieve(url, filename)
    if do_unzip:
        unzip(filename, './')
        # untar(filename, './')


def unzip(path2zip: str, path2unzip: str):
    """
    todo - fix, so that it works in any os
    :param path2zip:
    :param path2unzip:
    :return:
    """
    # print(f'{path2zip=}')
    # import zipfile
    # with zipfile.ZipFile(path2zip, 'r') as zip_ref:
    #     zip_ref.extractall(path2zip)
    path = str(Path(path2zip).expanduser())
    path2unzip = str(Path(path2unzip).expanduser())
    os.system(
        f'tar -xvzf {path2zip} -C {path2unzip}/')  # extract data set in above location i.e at path / 'miniImagenet'
    os.remove(path2zip)


def untar(path2zip: str, path2unzip: str):
    path = str(Path(path2zip).expanduser())
    path2unzip = str(Path(path2unzip).expanduser())
    os.system(
        f'tar -xvzf {path2zip} -C {path2unzip}/')  # extract data set in above location i.e at path / 'miniImagenet'
    os.remove(path2zip)


def _unzip(filename: Union[str, Path], extract_dir):
    """
    todo fix... perhaps not...?
    https://stackoverflow.com/questions/3451111/unzipping-files-in-python
    """
    # filename = str(filename)
    # print(f'unzipping {filename}...')
    # if os.path.exists(filename[:-7]):
    #     # remove = input(filename[:-7] + ' already exists. Do you want to remove it? (y/N)').lower()
    #     remove = 'y'
    #     if remove == 'y':
    #         execute('rm -r ' + filename[:-7])
    #     else:
    #         print('aborting..')
    #         sys.exit(-1)
    #
    # import shutil
    # shutil.unpack_archive(filename, extract_dir)
    #
    # execute(f'tar -xvzf {filename}')
    # print(f'done unzipping {filename}\n')
    pass


def save_ckpt(args: Namespace, mdl: nn.Module, optimizer: torch.optim.Optimizer,
              dirname: Union[None, Path] = None, ckpt_name: str = 'ckpt.pt'):
    """
    Saves checkpoint for any worker.
    Intended use is to save by worker that got a val loss that improved.


    """
    import dill
    import uutils

    dirname = args.log_root if (dirname is None) else dirname
    # - pickle ckpt
    assert uutils.xor(args.training_mode == 'epochs', args.training_mode == 'iterations')
    pickable_args = uutils.make_args_pickable(args)
    torch.save({'state_dict': mdl.state_dict(),
                'epoch_num': args.epoch_num,
                'it': args.it,
                'optimizer': optimizer.state_dict(),
                'args': pickable_args,
                'mdl': mdl},
               pickle_module=dill,
               f=dirname / ckpt_name)  # f'mdl_{epoch_num:03}.pt'


def equal_two_few_shot_cnn_models(model1: nn.Module, model2: nn.Module) -> bool:
    """
    Checks the two models have the same arch by comparing the string value of the module of each layer.
    Skips the cls layer since in sl vs ml, the final layer are of different sizes e.g. for maml its 5
    and for sl its 64 since sl trains with union of all layers.
    """
    # simple compare the feature layers, see file /Users/brando/automl-meta-learning/automl-proj-src/meta_learning/base_models/learner_from_opt_as_few_shot_paper.py
    # to see the deatails of what the model arch is and why we do it this way
    return str(model1.model.features) == str(model2.model.features)


# def get_layer_names_for_sim_analysis_5cnn(args: Namespace,
#                                           model: nn.Module,
#                                           layer_for_analysis: str,
#                                           include_final_layer_in_lst: bool = True) -> list[str]:
#     """
#     Do rep analysis on pool layer.
#
#     Thoughts:
#     This might be good since a layer for 5CNN looks as follows C,N,ReLU,P so the translational invariance has
#     been done, the non-linearity has been applied but due to pooling it's not going to be higher than expected
#     due to lots of zeros. It seems that makes sense, at least if we want the first layer to be more than just
#     a linear transform.
#     But worth trying all the layers.
#     """
#     layer_names: list = []
#     for name, m in model.named_modules():
#         # - to do analysis on inner activations
#         if layer_for_analysis in name:
#             layer_names.append(name)
#         assert name != 'spp', f'Get an spp layer, we are currently not training any model to handle spp.'
#         if include_final_layer_in_lst and name == 'cls':
#             layer_names.append(name)
#     # - print layers & return
#     from uutils.torch_uu.distributed import is_lead_worker
#     if is_lead_worker(args.rank):
#         print(layer_names)
#     return layer_names

def get_layer_names_to_do_sim_analysis_relu(args: Namespace, include_final_layer_in_lst: bool = True) -> list[str]:
    """
    Get the layers to do the similarity analysis.
    By default always include the last layer because it's the users job to either exclude it when doing a layer-wise
    average or removing it or otherwise.

    :param args:
    :param include_final_layer_in_lst:
    :return:
    """
    from uutils.torch_uu.distributed import is_lead_worker
    layer_names: list = []
    for name, m in args.meta_learner.base_model.named_modules():
        # - to do analysis on (non-linear) activations
        if 'relu' in name:
            layer_names.append(name)
        # - to do analysis on final layer
        if include_final_layer_in_lst:
            if 'fc4_final_l2' in name:
                layer_names.append(name)
    if is_lead_worker(args.rank):
        print(layer_names)
    return layer_names


def get_layer_names_to_do_sim_analysis_bn(args: Namespace, include_final_layer_in_lst: bool = True) -> list[str]:
    """
    Get the layers to do the similarity analysis.
    By default always include the last layer because it's the users job to either exclude it when doing a layer-wise
    average or removing it or otherwise.

    :param args:
    :param include_final_layer_in_lst:
    :return:
    """
    from uutils.torch_uu.distributed import is_lead_worker
    layer_names: list = []
    for name, m in args.meta_learner.base_model.named_modules():
        # - to do analysis on (non-linear) activations
        if 'bn' in name:
            layer_names.append(name)
        # - to do analysis on final layer
        if include_final_layer_in_lst:
            if 'fc4_final_l2' in name:
                layer_names.append(name)
    if is_lead_worker(args.rank):
        print(layer_names)
    return layer_names


def get_layer_names_to_do_sim_analysis_fc(args: Namespace, include_final_layer_in_lst: bool = True) -> list[str]:
    """
    Get the layers to do the similarity analysis.
    By default always include the last layer because it's the users job to either exclude it when doing a layer-wise
    average or removing it or otherwise.

    :param args:
    :param include_final_layer_in_lst:
    :return:
    """
    from uutils.torch_uu.distributed import is_lead_worker
    layer_names: list = []
    for name, m in args.meta_learner.base_model.named_modules():
        # - to do analysis on (non-linear) activations
        if 'fc' in name:
            layer_names.append(name)
    # - remove final element if user specified it
    L = len(layer_names)
    if not include_final_layer_in_lst:
        for i, name in enumerate(layer_names):
            if 'fc4_final_l2' in name:
                layer_names.pop(i)
        assert len(layer_names) == L - 1
    if is_lead_worker(args.rank):
        print(layer_names)
    return layer_names


def summarize_similarities(args: Namespace, sims: dict) -> dict:
    """
    Summarize similarity stats by computing the true expecations we care about.
    In particular wrt to tasks (and query examples).

    Note: returns a new dict with only the metrics we care about.
    """
    summarized_sim: dict = {}
    T, L = sims['cca'].size()
    # -- all layer stats
    # - compute means
    mean_layer_wise_sim: dict = {}
    assert T == args.meta_batch_size_eval
    # [T, L] -> [L], compute expectation per task for each layer
    mean_layer_wise_sim['cca'] = sims['cca'].mean(dim=0)
    mean_layer_wise_sim['cka'] = sims['cka'].mean(dim=0)
    mean_layer_wise_sim['op'] = sims['op'].mean(dim=0)
    assert mean_layer_wise_sim['cca'].size() == torch.Size([L])
    # [T] -> [1], compute expectation per task
    mean_layer_wise_sim['nes'] = sims['nes'].mean(dim=[0, 2])
    mean_layer_wise_sim['nes_output'] = sims['nes_output'].mean()
    mean_layer_wise_sim['query_loss'] = sims['query_loss'].mean()
    assert mean_layer_wise_sim['nes_output'].size() == torch.Size([])
    # - compute stds
    std_layer_wise_sim: dict = {}
    assert T == args.meta_batch_size_eval
    # [T, L] -> [L], compute expectation per task for each layer
    std_layer_wise_sim['cca'] = sims['cca'].std(dim=0)
    std_layer_wise_sim['cka'] = sims['cka'].std(dim=0)
    std_layer_wise_sim['op'] = sims['op'].std(dim=0)
    assert std_layer_wise_sim['cca'].size() == torch.Size([L])
    # [T] -> [1], compute expectation per task
    std_layer_wise_sim['nes'] = sims['nes'].std(dim=[0, 2])
    std_layer_wise_sim['nes_output'] = sims['nes_output'].std()
    std_layer_wise_sim['query_loss'] = sims['query_loss'].std()
    assert std_layer_wise_sim['nes_output'].size() == torch.Size([])

    # -- rep stats
    mean_summarized_rep_sim: dict = {}
    std_summarized_rep_sim: dict = {}
    mean_summarized_rep_sim['cca'] = sims['cca'][:, :-1].mean()
    mean_summarized_rep_sim['cka'] = sims['cka'][:, :-1].mean()
    mean_summarized_rep_sim['op'] = sims['op'][:, :-1].mean()
    mean_summarized_rep_sim['nes'] = sims['nes'][:, :-1].mean()
    assert mean_summarized_rep_sim['cca'].size() == torch.Size([])
    std_summarized_rep_sim['cca'] = sims['cca'][:, :-1].std()
    std_summarized_rep_sim['cka'] = sims['cka'][:, :-1].std()
    std_summarized_rep_sim['op'] = sims['op'][:, :-1].std()
    std_summarized_rep_sim['nes'] = sims['nes'][:, :-1].std()
    assert std_summarized_rep_sim['cca'].size() == torch.Size([])
    return mean_layer_wise_sim, std_layer_wise_sim, mean_summarized_rep_sim, std_summarized_rep_sim


def log_sim_to_check_presence_of_feature_reuse_mdl1_vs_mdl2(args: Namespace,
                                                            it: int,
                                                            mdl1: nn.Module, mdl2: nn.Module,
                                                            batch_x: torch.Tensor, batch_y: torch.Tensor,

                                                            # spt_x, spt_y, qry_x, qry_y,  # these are multiple tasks

                                                            log_freq_for_detection_of_feature_reuse: int = 3,

                                                            force_log: bool = False,
                                                            parallel: bool = False,
                                                            iter_tasks=None,
                                                            log_to_wandb: bool = False,
                                                            show_layerwise_sims: bool = True
                                                            ):
    """
    Goal is to see if similarity is small s <<< 0.9 (at least s < 0.8) since this suggests that
    """
    import wandb
    import uutils.torch_uu as torch_uu
    from pprint import pprint
    from uutils.torch_uu import summarize_similarities
    from uutils.torch_uu.distributed import is_lead_worker
    # - is it epoch or iteration
    it_or_epoch: str = 'epoch_num' if args.training_mode == 'epochs' else 'it'
    sim_or_dist: str = 'sim'
    if hasattr(args, 'metrics_as_dist'):
        sim_or_dist: str = 'dist' if args.metrics_as_dist else sim_or_dist
    total_its: int = args.num_empochs if args.training_mode == 'epochs' else args.num_its

    if (it == total_its - 1 or force_log) and is_lead_worker(args.rank):
        # if (it % log_freq_for_detection_of_feature_reuse == 0 or it == total_its - 1 or force_log) and is_lead_worker(args.rank):
        #     if hasattr(args, 'metrics_as_dist'):
        #         sims = args.meta_learner.compute_functional_similarities(spt_x, spt_y, qry_x, qry_y, args.layer_names, parallel=parallel, iter_tasks=iter_tasks, metric_as_dist=args.metrics_as_dist)
        #     else:
        #         sims = args.meta_learner.compute_functional_similarities(spt_x, spt_y, qry_x, qry_y, args.layer_names, parallel=parallel, iter_tasks=iter_tasks)
        sims = distances_btw_models(args, mdl1, mdl2, batch_x, batch_y, args.layer_names, args.metrics_as_dist)
        print(sims)
        # mean_layer_wise_sim, std_layer_wise_sim, mean_summarized_rep_sim, std_summarized_rep_sim = summarize_similarities(args, sims)

        # -- log (print)
        args.logger.log(f' \n------ {sim_or_dist} stats: {it_or_epoch}={it} ------')
        # - per layer
        # if show_layerwise_sims:
        print(f'---- Layer-Wise metrics ----')
        # print(f'mean_layer_wise_{sim_or_dist} (per layer)')
        # pprint(mean_layer_wise_sim)
        # print(f'std_layer_wise_{sim_or_dist} (per layer)')
        # pprint(std_layer_wise_sim)
        #
        # # - rep sim
        # print(f'---- Representation metrics ----')
        # print(f'mean_summarized_rep_{sim_or_dist} (summary for rep layer)')
        # pprint(mean_summarized_rep_sim)
        # print(f'std_summarized_rep_{sim_or_dist} (summary for rep layer)')
        # pprint(std_summarized_rep_sim)
        # args.logger.log(f' -- sim stats : {it_or_epoch}={it} --')

        # error bars with wandb: https://community.wandb.ai/t/how-does-one-plot-plots-with-error-bars/651
        # - log to wandb
        # if log_to_wandb:
        #     if it == 0:
        #         # have all metrics be tracked with it or epoch (custom step)
        #         #     wandb.define_metric(f'layer average {metric}', step_metric=it_or_epoch)
        #         for metric in mean_summarized_rep_sim.keys():
        #             wandb.define_metric(f'rep mean {metric}', step_metric=it_or_epoch)
        #     # wandb.log per layer
        #     rep_summary_log = {f'rep mean {metric}': sim for metric, sim in mean_summarized_rep_sim.items()}
        #     rep_summary_log[it_or_epoch] = it
        #     wandb.log(rep_summary_log, commit=True)


def distances_btw_models(args: Namespace,
                         model1: nn.Module, model2: nn.Module,
                         batch_x: torch.Tensor, batch_y: torch.Tensor,
                         layer_names: list[str],
                         metrics_as_dist: bool = True) -> dict:
    """
    Compute the distance/sim between two models give a batch of example (this assumes there are no tasks involved, just
    two batch of any type of examples).
    """
    L: int = len(layer_names)
    # make into eval
    model1.eval()
    model2.eval()
    # -- compute sims
    x: torch.Tensor = batch_x
    if torch.cuda.is_available():
        x = x.cuda()
    # - compute cca sims
    cca: list[float] = get_cxa_similarities_per_layer(model1, model2, x, layer_names, sim_type='pwcca')
    cka: list[float] = get_cxa_similarities_per_layer(model1, model2, x, layer_names, sim_type='lincka')
    assert len(cca) == L
    assert len(cka) == L
    # -- get l2 sims per layer
    # op = get_l2_similarities_per_layer(model1, model2, x, layer_names, sim_type='op_torch')
    # nes = get_l2_similarities_per_layer(model1, model2, x, layer_names, sim_type='nes_torch')
    # cosine = get_l2_similarities_per_layer(model1, model2, x, layer_names, sim_type='cosine_torch')

    # y = self.base_model(qry_x_t)
    # y_adapt = fmodel(qry_x_t)
    # # dim=0 because we have single numbers and we are taking the NES in the batch direction
    # nes_output = uutils.torch_uu.nes_torch(y.squeeze(), y_adapt.squeeze(), dim=0).item()
    #
    # query_loss = self.args.criterion(y, y_adapt).item()
    # # sims = [cca, cka, nes, cosine, nes_output, query_loss]
    # -- from [Tasks, Sims] -> [Sims, Tasks]
    # sims = {'cca': [], 'cka': [], 'op': [],  # [L]
    #         'nes': [], 'cosine': [],  # [L, K]
    #         'nes_output': [], 'query_loss': []  # [1]
    #         }
    # sims = {'cca': cca, 'cka': cka, 'op': op,  # [L]
    #         'nes': nes, 'cosine': cosine,  # [L, K]
    #         'nes_output': nes_output, 'query_loss': query_loss  # [1]
    #         }
    # sims = {metric: tensorify(sim).detach() for metric, sim in sims.items()}
    out_metrics: dict = {}
    for metric, sim in sims.items():
        out_metrics[metric] = tensorify(sim).detach()
        if metrics_as_dist and metric != 'query_loss':
            out_metrics[metric] = 1.0 - out_metrics[metric]
            if metric != 'cosine':
                error_tolerance: float = -0.0001
                assert (out_metrics[
                            metric] >= error_tolerance).all(), f'Distances are positive but got a negative value somewhere for metric {metric=}.'
    return out_metrics


def get_cxa_similarities_per_layer(model1: nn.Module, model2: nn.Module,
                                   x: torch.Tensor, layer_names: list[str],
                                   sim_type: str = 'pwcca'):
    """
    Get [..., s_l, ...] cca sim per layer (for this data set)
    """
    from uutils.torch_uu import cxa_sim
    sims_per_layer = []
    for layer_name in layer_names:
        # sim = cxa_sim(model1, model2, x, layer_name, cca_size=self.args.cca_size, iters=1, cxa_sim_type=sim_type)
        sim = cxa_sim(model1, model2, x, layer_name, iters=1, cxa_sim_type=sim_type)
        sims_per_layer.append(sim)
    return sims_per_layer  # [..., s_l, ...]_l


def compare_based_on_mdl1_vs_mdl2(args: Namespace, meta_dataloader):
    print(f'{args.num_workers=}')
    # print(f'-->{args.meta_batch_size_eval=}')
    print(f'-->{args.num_its=}')
    # print(f'-->{args.nb_inner_train_steps=}')
    print(f'-->{args.metrics_as_dist=}')
    #
    # bar_it = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    args.it = 1
    halt: bool = False
    while not halt:
        # spt_x, spt_y, qry_x, qry_y = next(meta_dataloader)
        for batch_idx, batch in enumerate(meta_dataloader):
            print(f'it = {args.it}')
            spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
            batch_x, batch_y = qry_x[0], qry_y[0]

            # args.model1(batch_x)
            # args.model2(batch_x)

            # meta_eval_loss, meta_eval_acc = args.meta_learner(spt_x, spt_y, qry_x, qry_y)
            # - todo, get the loss, accuracies of both models first
            # args.model1

            # -- log it stats
            log_sim_to_check_presence_of_feature_reuse_mdl1_vs_mdl2(args, args.it, args.model1, args.model2, batch_x,
                                                                    batch_y, force_log=True,
                                                                    parallel=args.sim_compute_parallel)

            # - break
            halt: bool = args.it >= args.num_its - 1
            if halt:
                break
            args.it += 1


def compare_based_on_meta_learner(args: Namespace, meta_dataloader):
    print(f'{args.num_workers=}')
    print(f'-->{args.meta_batch_size_eval=}')
    print(f'-->{args.num_its=}')
    print(f'-->{args.nb_inner_train_steps=}')
    print(f'-->{args.metrics_as_dist=}')
    # bar_it = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    args.it = 1
    halt: bool = False
    while not halt:
        for batch_idx, batch in enumerate(meta_dataloader):
            print(f'it = {args.it}')
            spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
            batch_x, batch_y = qry_x[0], qry_y[0]

            meta_eval_loss, meta_eval_acc = args.meta_learner(spt_x, spt_y, qry_x, qry_y)

            # -- log it stats
            # log_sim_to_check_presence_of_feature_reuse(args, args.it, spt_x, spt_y, qry_x, qry_y, force_log=True, parallel=args.sim_compute_parallel, show_layerwise_sims=args.show_layerwise_sims)
            log_sim_to_check_presence_of_feature_reuse_mdl1_vs_mdl2()

            # - break
            halt: bool = args.it >= args.num_its - 1
            if halt:
                break
            args.it += 1
    return meta_eval_loss, meta_eval_acc


# def get_sim_vs_num_data(args: Namespace, mdl1: nn.Module, mdl2: nn.Module,
#                         X1: Tensor, X2: Tensor,
#                         layer_name: str, cxa_dist_type: str) -> tuple[list[int], list[float]]:
#     """
#     Plots sim vs N given a fixed D.
#
#     X1: [n_c*k_eval*H*W, F]
#     """
#     assert (X1.size(0) == X2.size(0)), f'Data sets must to have the same sizes for CCA type analysis to work.' \
#                                        f'but got: {X1.size(0)=}, {X2.size(0)=}'
#     # - get the sims vs data_set sizes
#     # data_sizes: list[int] = [X1.size(0)]
#     # data_sizes: list[int] = [10]
#     data_sizes: list[int] = [args.k_eval * args.n_classes]
#     print(f'# examples = {args.k_eval * args.n_classes}')
#     # data_sizes: list[int] = [10, 25, 50, 100, 101, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 100_000]
#     sims: list[float] = []
#     for b in data_sizes:
#         x1, x2 = X1[:b], X2[:b]  # get first b images
#         sim: float = cxa_sim_general(mdl1, mdl2, x1, x2, layer_name, downsample_size=None, iters=1,
#                                      cxa_dist_type=cxa_dist_type)
#         # sim: float = cxa_sim_general(mdl1, mdl2, x1, x2, layer_name, downsample_size=2, iters=1, cxa_dist_type=cxa_dist_type)
#         sims.append(sim)
#     return data_sizes, sims


def assert_sim_of_model_with_itself_is_approx_one(mdl: nn.Module, X: Tensor,
                                                  layer_name: str,
                                                  metric_comparison_type: str = 'pwcca',
                                                  metric_as_sim_or_dist: str = 'dist') -> bool:
    """
    Returns true if model is ok. If not it asserts against you (never returns False).
    """
    dist: float = get_metric(mdl, mdl, X, X, layer_name, metric_comparison_type=metric_comparison_type,
                             metric_as_sim_or_dist=metric_as_sim_or_dist)
    print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
    assert approx_equal(dist, 0.0), f'Sim should be close to 1.0 but got: {dist=}'
    return True


# -- pytorch hooks
# ref: https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904

def module_hook(module: nn.Module, input: Tensor, output: Tensor):
    """
    For nn.Module objects only.
    """
    pass


def tensor_hook(grad: Tensor):
    """
    For Tensor objects only.
    Only executed during the *backward* pass!
    """
    pass


# For Tensor objects only.
# Only executed during the *backward* pass!

def hook_for_printing_output_shape(layer: nn.Module, input: Tensor, output: Tensor):
    """
    PyTorch hook for printing the output shape.

    The idea is to register this hook for each module, wrap your model so that it registers this hook to each
    of it's modules then use this automatically when calling the forward pass of your model.
    """
    print(f"{layer.__name__}: {output.shape}")


class VerboseExecutionHook(nn.Module):
    """
    Registers a hook that when the model is ran, it prints the output share for each layer
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(hook_for_printing_output_shape)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class FeatureExtractorHook(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model: nn.Module = model
        self.layers: Iterable[str] = layers
        self._features: dict[str, Tensor] = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            # add hook that saves the features to the current hook obj for the model
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> callable:
        # the trick is that the fn has the self (which is the hook self) hardcoded in the passable function,
        # so even though it only takes module, input, output, you can collect the features.
        def fn(_, __, output):
            # append output features of current layer to hook_self obj.
            self._features[layer_id] = output

        return fn

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Returns the features for each layer for this model in a dict with the format:
            features = {layer: str -> tesnor_for_layer: torchTensor}
        i.e. each string for the layer maps to the feature tensor for that layer.
        """
        _ = self.model(x)
        return self._features


def gradient_clipper_hook(model: nn.Module, val: float) -> nn.Module:
    for parameter in model.parameters():
        parameter.register_hook(lambda grad: grad.clamp_(-val, val))

    return model


class GetMaxFiltersExtractorHook(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model: nn.Module = model
        self.layers: Iterable[str] = layers
        self._features: dict[str, Tensor] = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            # add hook that saves the features to the current hook obj for the model
            hook_for_collecting_features: callable = self.get_save_outputs_hook(layer_id)
            layer.register_forward_hook(hook_for_collecting_features)

    def get_save_outputs_hook(self, layer_id: str) -> callable:
        # the trick is that the fn has the self (which is the hook self) hardcoded in the passable function,
        # so even though it only takes module, input, output, you can collect the features.
        def hook(_, __, output):
            # append output features of current layer to hook_self obj.
            self._features[layer_id] = output

        return hook

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Returns the features for each layer for this model in a dict with the format:
            features = {layer: str -> tesnor_for_layer: torchTensor}
        i.e. each string for the layer maps to the feature tensor for that layer.
        """
        _ = self.model(x)
        return self._features


# -- misc

def remove_hook(mdl: nn.Module, hook):
    """
    ref: https://github.com/pytorch/pytorch/issues/5037
    """
    handle = mdl.register_forward_hook(hook)
    handle.remove()


def approx_equal(val1: float, val2: float, tolerance: float = 1.0e-4) -> bool:
    """
    Returns wether two values are approximately equal e.g. if they are less than 4 orders of magnitude apart.
    """
    eq: bool = abs(val1 - val2) <= tolerance
    return eq


def get_identity_data(B: int) -> torch.Tensor:
    """
    Returns an identity data of size [B, D] = [B, B] so that the data doesn't affect the forward propagation (or at
    least as little as possible).

    Note: the dim/num neurons/features D is equal to the batch size B.
    """
    data: torch.Tensor = torch.diag(torch.ones(B))
    return data


def get_normal_data(B: int, Din: int, loc: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    """
    Returns normally distributed data of size [B, Din].

    Note: torch.randn(B, D) does something similar.
    """
    data: torch.Tensor = torch.distributions.Normal(loc=loc, scale=scale).sample((B, Din))
    return data


# --

# -- tests

def ned_test():
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
        # print(ned_torch(x1, x2, dim=dim))


def tensorify_test():
    t = [1, 2, 3]
    print(tensorify(t).size())
    tt = [t, t, t]
    print(tensorify(tt))
    ttt = [tt, tt, tt]
    print(tensorify(ttt))


def compressed_r2_score():
    y = torch.randn(10, 1)
    y_pred = 2 * y
    c_r2 = compressed_r2_score(y, y_pred)
    c_r2_torch = compressed_r2_score_from_torch(y, y_pred)
    assert (c_r2_torch == c_r2)


def topk_accuracy_and_accuracy_test():
    import torch
    import torch.nn as nn

    in_features = 32
    n_classes = 10
    batch_size = 1024

    mdl = nn.Linear(in_features=in_features, out_features=n_classes)

    x = torch.randn(batch_size, in_features)
    y_logits = mdl(x)
    y = torch.randint(high=n_classes, size=(batch_size,))

    acc_top1, acc_top2, acc_top5 = accuracy(output=y_logits, target=y, topk=(1, 2, 5))
    acc_top1_, acc_top2_, acc_top5_ = topk_accuracy(output=y_logits, target=y, topk=(1, 2, 5))
    assert (acc_top5 == acc_top5_)
    assert (acc_top1 == acc_top1_)
    acc1 = calc_accuracy(mdl, x, y)
    acc1_ = calc_accuracy_from_logits(y_logits, y)
    assert (acc1 == acc1_)
    assert (acc1_ == acc_top1)


def split_test():
    files = list(range(10))
    train, test = split_two(files)
    print(train, test)
    train, val, test = split_three(files)
    print(train, val, test)


def split_data_train_val_test():
    from sklearn.model_selection import train_test_split

    # overall split 85:10:5

    X = list(range(100))
    y = list(range(len(X)))

    # first do 85:15 then do 2:1 for val split
    # its ok to set it to False since its ok to shuffle but then allow reproducibility with random_state
    # shuffle = False  # shufflebool, default=True, Whether or not to shuffle the data_lib before splitting. If shuffle=False then stratify must be None.
    random_state = 1  # Controls the shuffling applied to the data_lib before applying the split. Pass an int for reproducible output across multiple function calls.
    test_size = 0.15
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(len(X_train))
    print(len(X_val_test))

    # then 2/3 for val, 1/3 for test to get 10:5 split
    test_size = 1.0 / 3.0
    X_val, X_test, y_test, y_test = train_test_split(X_val_test, y_val_test, test_size=test_size,
                                                     random_state=random_state)
    print(len(X_val))
    print(len(X_test))


def simple_determinism_test():
    args = Namespace(seed=0, deterministic_alg=True)
    make_code_deterministic(args.seed, args.deterministic_alg)
    #
    x = torch.randn(3, 3, 3)
    print(f'{x.sum()=}')
    out = x @ x
    print(f'{out.sum()}')


def op_test():
    from uutils.torch_uu.models import hardcoded_3_layer_model

    force = True
    # force = False
    mdl1 = hardcoded_3_layer_model(5, 1)
    mdl2 = hardcoded_3_layer_model(5, 1)
    batch_size = 4
    X = torch.randn(batch_size, 5)
    import copy
    from uutils.torch_uu import l2_sim_torch
    # get [..., s_l, ...] sim per layer (for this data set)
    modules = zip(mdl1.named_children(), mdl2.named_children())
    sims_per_layer = []
    out1 = X
    out2 = X
    for (name1, m1), (name2, m2) in modules:
        # if name1 in layer_names:
        if 'ReLU' in name1 or force:  # only compute on activation
            out1 = m1(out1)
            m2_callable = copy.deepcopy(m1)
            m2_callable.load_state_dict(m2.state_dict())
            out2 = m2_callable(out2)
            sim = l2_sim_torch(out1, out2, sim_type='op_torch')
            sims_per_layer.append((name1, sim))
    pprint(sims_per_layer)


def anatome_test_are_same_nets_very_similar():
    """
    Same model with same data even if down sampled, should show similar nets.
    """
    from uutils.torch_uu.models import hardcoded_3_layer_model
    B = 1024
    Din = 524
    downsample_size = 4
    Dout = Din
    mdl1 = hardcoded_3_layer_model(Din, Dout)
    mdl2 = mdl1
    # - layer name
    # layer_name = 'fc0'
    # layer_name = 'fc1'
    layer_name = 'fc2'
    # - data
    X: torch.Tensor = torch.distributions.Uniform(low=-1, high=1).sample((B, Din))
    scca_full: float = sCXA(mdl1, mdl2, X, layer_name, downsample_size=None)
    assert (abs(scca_full - 1.0) < 1e-5)
    scca_downsampled: float = sCXA(mdl1, mdl2, X, layer_name, downsample_size=downsample_size)
    assert (abs(scca_downsampled - 1.0) < 1e-5)


def anatome_test_are_random_vs_pretrain_resnets_different():
    """
    random vs pre-trained nets should show different nets
    - no downsample
    - still true if downsample (but perhaps similarity increases, due to collapsing nets makes r.v.s
    interact more btw each other, so correlation is expected to increase).
    """
    from torchvision.models import resnet18
    B = 1024
    C, H, W = 3, 64, 64
    print(f'Din ~ {(C*H*W)=}')
    downsample_size = 4
    mdl1 = resnet18()
    mdl2 = resnet18(pretrained=True)
    # - layer name
    # layer_name = 'bn1'
    # layer_name = 'layer1.0.bn2'
    # layer_name = 'layer2.1.bn2'
    layer_name = 'layer4.1.bn2'
    # layer_name = 'fc'
    print(f'{layer_name=}')

    # # -- we expect low CCA/sim since random nets vs pre-trained nets are different (especially on real data)
    # # - random data test
    X: torch.Tensor = torch.distributions.Uniform(low=-1, high=1).sample((B, C, H, W))
    scca_full_random_data: float = sCXA(mdl1, mdl2, X, layer_name, downsample_size=None, cxa_dist_type='pwcca')
    # scca_full_random_data: float = sCXA(mdl1, mdl2, X, layer_name, downsample_size=None, cxa_dist_type='lincka')
    # scca_full_random_data: float = sCXA(mdl1, mdl2, X, layer_name, downsample_size=None, cxa_dist_type='svcca')
    print(f'Are random net & pre-trained net similar? They should not (so sim should be small):\n'
          f'-> {scca_full_random_data=} (but might be more similar than expected on random data)')
    # scca_downsampled: float = sCXA(mdl1, mdl2, X, layer_name, downsample_size=downsample_size)
    # print(f'Are random net & pre-trained net similar? They should not (so sim should be small): {scca_downsampled=}')

    #
    mdl1 = resnet18()
    mdl2 = resnet18(pretrained=True)
    # - mini-imagenet test (the difference should be accentuated btw random net & pre-trained on real img data)
    from uutils.torch_uu.dataloaders import get_set_of_examples_from_mini_imagenet
    B = 512
    k_eval: int = B  # num examples is about M = k_eval*(num_classes) = B*(num_classes)
    X: torch.Tensor = get_set_of_examples_from_mini_imagenet(k_eval)
    scca_full_mini_imagenet_data: float = sCXA(mdl1, mdl2, X, layer_name, downsample_size=None)
    scca_full_random_data: float = sCXA(mdl1, mdl2, X, layer_name, downsample_size=None, cxa_dist_type='pwcca')
    # scca_full_random_data: float = sCXA(mdl1, mdl2, X, layer_name, downsample_size=None, cxa_dist_type='lincka')
    # scca_full_random_data: float = sCXA(mdl1, mdl2, X, layer_name, downsample_size=None, cxa_dist_type='svcca')
    print(f'Are random net & pre-trained net similar? They should not (so sim should be small):\n'
          f'{scca_full_mini_imagenet_data=} (the difference should be accentuated with real data, so lowest sim)')
    print()
    # assert(scca_full_mini_imagenet_data < scca_full_random_data), f'Sim should decrease, because the pre-trained net' \
    #                                                               f'was trained on real images, so the weights are' \
    #                                                               f'tuned for it but random weights are not, which' \
    #                                                               f'should increase the difference so the sim' \
    #                                                               f'should be lowest here. i.e. we want ' \
    #                                                               f'{scca_full_mini_imagenet_data}<{scca_full_random_data}'
    # scca_downsampled: float = sCXA(mdl1, mdl2, X, layer_name, downsample_size=downsample_size)
    # print(f'Are random net & pre-trained net similar? They should not (so sim should be small): {scca_downsampled=}')


def anatome_test_what_happens_when_downsampling_increases_do_nets_get_more_similar_or_different():
    """
    - real, fake data
    - focus on pre-trained net since that is what I am comparing stuff during my research, not random nets.
    """
    pass


def cov(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
    """
    Compute covariance of input

    :param x: [M, D]
    :param y: [M, D]
    :return:
    """
    if y is not None:
        y = x
    else:
        assert x.size(0) == y.size(0)
    # - center first
    x = center(x, dim=0)
    y = center(y, dim=0)
    # - conv = E[XY] is outer product of X^T Y or X Y^T depending on shapes
    sigma_xy: Tensor = x.T @ y
    return sigma_xy


# -- tests

def verbose_exec_test():
    import torch
    from torchvision.models import resnet50

    verbose_resnet = VerboseExecutionHook(resnet50())
    dummy_input = torch.ones(10, 3, 224, 224)

    _ = verbose_resnet(dummy_input)
    # conv1: torch.Size([10, 64, 112, 112])
    # bn1: torch.Size([10, 64, 112, 112])
    # relu: torch.Size([10, 64, 112, 112])
    # maxpool: torch.Size([10, 64, 56, 56])
    # layer1: torch.Size([10, 256, 56, 56])
    # layer2: torch.Size([10, 512, 28, 28])
    # layer3: torch.Size([10, 1024, 14, 14])
    # layer4: torch.Size([10, 2048, 7, 7])
    # avgpool: torch.Size([10, 2048, 1, 1])
    # fc: torch.Size([10, 1000])


def feature_extractor_hook_test():
    import torch
    from torchvision.models import resnet50

    resnet_features = FeatureExtractorHook(resnet50(), layers=["layer4", "avgpool"])
    dummy_input = torch.ones(10, 3, 224, 224)
    features = resnet_features(dummy_input)

    print({name: output.shape for name, output in features.items()})
    # {'layer4': torch.Size([10, 2048, 7, 7]), 'avgpool': torch.Size([10, 2048, 1, 1])}


def grad_clipper_hook_test():
    import torch
    from torchvision.models import resnet50

    dummy_input = torch.ones(10, 3, 224, 224)
    clipped_resnet = gradient_clipper_hook(resnet50(), 0.01)
    pred = clipped_resnet(dummy_input)
    loss = pred.log().mean()
    loss.backward()

    print(clipped_resnet.fc.bias.grad[:25])


def _resume_from_checkpoint_meta_learning_for_resnets_rfs_test():
    import uutils
    from uutils.torch_uu.models import reset_all_weights
    import copy
    # - get args to ckpt
    args: Namespace = Namespace()
    args.path_to_checkpoint: str = '~/data_folder_fall2020_spring2021/logs/nov_all_mini_imagenet_expts/logs_Nov05_15-44-03_jobid_668'
    args: Namespace = uutils.make_args_from_metalearning_checkpoint(args=args,
                                                                    path2args=args.path_to_checkpoint,
                                                                    filename='args.json',
                                                                    precedence_to_args_checkpoint=True,
                                                                    it=37_500)
    args: Namespace = uutils.setup_args_for_experiment(args)
    # - get model from ckpt
    mdl_ckpt, outer_opt, meta_learner = get_model_opt_meta_learner_to_resume_checkpoint_resnets_rfs(args,
                                                                                                    path2ckpt=args.path_to_checkpoint,
                                                                                                    filename='ckpt_file.pt')
    # - f_rand
    mdl_rand = copy.deepcopy(mdl_ckpt)
    reset_all_weights(mdl_rand)
    # - print if ckpt model is different from a random model
    print(lp_norm(mdl_ckpt))
    print(lp_norm(mdl_rand))
    assert(lp_norm(mdl_ckpt) != lp_norm(mdl_rand))


# -- _main

if __name__ == '__main__':
    # test_ned()
    # test_tensorify()
    # test_compressed_r2_score()
    # test_topk_accuracy_and_accuracy()
    # test_simple_determinism()
    # op_test()
    # anatome_test_are_same_nets_very_similar()
    # anatome_test_are_random_vs_pretrain_resnets_different()
    # verbose_exec_test()
    # feature_extractor_hook_test()
    _resume_from_checkpoint_meta_learning_for_resnets_rfs_test()
    print('Done\a')
