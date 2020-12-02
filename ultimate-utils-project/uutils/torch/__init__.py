'''
Torch Based Utils/universal methods

Utils class with useful helper functions

utils: https://www.quora.com/What-do-utils-files-tend-to-be-in-computer-programming-documentation

'''
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

def cca(mdl1, mdl2, meta_batch, layer_name, cca_size=8, iters=2):
    # meta_batch [T, N*K, CHW], [T, K, D]
    from anatome import SimilarityHook
    # get sim/dis functions
    hook1 = SimilarityHook(mdl1, layer_name)
    hook2 = SimilarityHook(mdl2, layer_name)
    mdl1.eval()
    mdl2.eval()
    for _ in range(iters):  # might make sense to go through multiple is NN is stochastic e.g. BN, dropout layers
        # x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
        # x = torch.torch.distributions.Uniform(low=-1, high=1).sample((15, 1))
        # x = torch.torch.distributions.Uniform(low=-1, high=1).sample((500, 1))
        x = meta_batch
        mdl1(x)
        mdl2(x)
    dist = hook1.distance(hook2, size=cca_size)
    return dist

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

    ned = 0.5*np.var(x - y) / (np.var(x) + np.var(y)) = 0.5 variance of difference / total variance individually

    reference: https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance

    @param x:
    @param y:
    @return:
    """
    ned = 0.5*np.var(f - y) / (np.var(f) + np.var(y))
    return ned

def r2_symmetric(f, y, r2_type='explained_variance'):
    """
    Normalized (symmetric) R^2 with respect to two vectors:



    reference:
    - https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance
    - https://en.wikipedia.org/wiki/Coefficient_of_determination#:~:text=R2%20is%20a%20statistic,predictions%20perfectly%20fit%20the%20data.
    - https://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score
    - https://en.wikipedia.org/wiki/Fraction_of_variance_unexplained
    - https://en.wikipedia.org/wiki/Explained_variation

    @param x:
    @param y:
    @return:
    """
    # import sklearn.metrics.explained_variance_score as evar
    from sklearn.metrics import mean_squared_error as mse
    f = f if type(f) != torch.Tensor else f.detach().cpu().numpy()
    y = y if type(y) != torch.Tensor else y.detach().cpu().numpy()
    if r2_type == 'my_explained_variance':
        # evar_f2y
        # evar_y2f
        # r2 = (evar_f2y + evar_y2f) / (np.var(f) + np.var(y))
        raise ValueError(f'Not implemented: {r2_type}')
    elif r2_type == '1_minus_total_residuals':
        r2 = 1 - ((2 * mse(f, y)) / (np.var(f) + np.var(y)))
    elif r2_type == 'ned':
        return ned(f, y)
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

#######

def test():
    print()


if __name__ == '__main__':
    pass
