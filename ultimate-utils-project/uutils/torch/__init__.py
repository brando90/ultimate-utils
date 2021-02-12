'''
Torch Based Utils/universal methods

Utils class with useful helper functions

utils: https://www.quora.com/What-do-utils-files-tend-to-be-in-computer-programming-documentation

'''
from datetime import datetime

import torch
import torch.nn as nn

import numpy as np
import scipy.integrate as integrate

from collections import OrderedDict

import dill as pickle

import os

from pathlib import Path

import copy

from pdb import set_trace as st

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def helloworld():
    print('hello world torch_utils!')

# meta-optimizer utils

def set_requires_grad(bool, mdl):
    for name, w in mdl.named_parameters():
        w.requires_grad = bool

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

def calc_error(mdl, X, Y):
    # acc == (true != mdl(x).max(1).item() / true.size(0)
    train_acc = calc_accuracy(mdl, X, Y)
    train_err = 1.0 - train_acc
    return train_err

def calc_accuracy(mdl, X, Y):
    # reduce/collapse the classification dimension according to max op (most likely label according to model)
    # resulting in most likely label
    max_vals, max_indices = mdl(X).max(1)
    # assumes the 0th dimension is batch size
    n = max_indices.size(0)
    # calulate acc (note .item() to do float division)
    acc = (max_indices == Y).sum().item() / n
    return acc

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
    ## https://discuss.pytorch.org/t/advantages-disadvantages-of-using-pickle-module-to-save-models-vs-torch-save/79016
    ## make dir to logs (and ckpts) if not present. Throw no exceptions if it already exists
    path_to_ckpt = args.logger.current_logs_path
    path_to_ckpt.mkdir(parents=True, exist_ok=True) # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.
    ckpt_path_plus_path = path_to_ckpt / Path('db')

    ## Pickle args & logger (note logger is inside args already), source: https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
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
        ## combine new args with old args
        args.base_model = "no child mdl in args see meta_learner"
        args = args_recovered
        return args, meta_learner

def test_ckpt_meta_learning(args, meta_learner, verbose=False):
    path_to_ckpt = args.logger.current_logs_path
    path_to_ckpt.mkdir(parents=True, exist_ok=True) # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.
    ckpt_path_plus_path = path_to_ckpt / Path('db')

    ## Pickle args & logger (note logger is inside args already), source: https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
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

##

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # st()
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # st()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res
##

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
    norm, abs_err = integrate.quad(pointwise_diff, lb, ub)
    return norm**(1/p), abs_err

def cxa_dist(mdl1, mdl2, meta_batch, layer_name, cca_size=2, iters=2, cxa_dist_type='pwcca'):
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
    if cxa_dist_type == 'lincka':
        sim = dist  # only true for cka for this library
        dist = 1 - sim  # since dist = 1 - sim but cka is outputing in this library the sim
    return dist

def cxa_sim(mdl1, mdl2, meta_batch, layer_name, cca_size=8, iters=2, cxa_sim_type='pwcca'):
    dist = cxa_dist(mdl1, mdl2, meta_batch, layer_name, cca_size, iters, cxa_sim_type)
    return 1 - dist

def cca_rand_data(mdl1, mdl2, num_samples_per_task, layer_name, lb=-1, ub=1, Din=1, cca_size=8, iters=2):
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

# def cca(mdl1, mdl2, meta_batch, layer_name, cca_size=8, iters=2):
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

# def cca(mdl1, mdl2, dataloader, cca_size=8, iters=10):
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

def ned_torch(x1, x2, dim=1, eps=1e-8):
    """
    Normalized eucledian distance in pytorch.

    For comparison of two vecs directly make sure vecs are of size [K]
    For comparison of two batch of 1D acs (e.g. scores) make sure it's of shape [B, 1]
    For the rest specify the dimension. Common use case [B, D] -> [B, 1] for comparing two set of
    activations of size D.

    https://discuss.pytorch.org/t/how-does-one-compute-the-normalized-euclidean-distance-similarity-in-a-numerically-stable-way-in-a-vectorized-way-in-pytorch/110829
    https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance/498753?noredirect=1#comment937825_498753
    """
    # to compute ned for two individual vectors e.g to compute a loss (NOT BATCHES/COLLECTIONS of vectorsc)
    if len(x1.size()) == 1:
        # [K] -> [1]
        ned_2 = 0.5 * ((x1 - x2).var() / (x1.var() + x2.var() + eps))
    # if the input is a (row) vector e.g. when comparing two batches of acts of D=1 like with scores right before sf
    elif x1.size(1) == 1:  # note this special case is needed since var over dim=1 in this case is nan.
        # [B, 1] -> [B, 1]
        ned_2 = 0.5 * ((x1 - x2)**2 / (x1**2 + x2**2 + eps))
    # common case is if input is a batch
    else:
        # e.g. [B, D] -> [B, 1]
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
            # compute average cxa per layer: [T, L, K_eval] -> [L]
            avg_sims = tensor_of_metrics.mean(dim=[0, 2])
            std_sims = tensor_of_metrics.std(dim=[0, 2])
            # compute representation & all avg cxa [T, L, K_eval] -> [1]
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

if __name__ == '__main__':
    test_ned()
    # test_tensorify()
    print('Done\a')
