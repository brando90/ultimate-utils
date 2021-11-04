import gc
from argparse import Namespace
from collections import OrderedDict
from typing import Optional

import higher
import numpy as np
from higher.optim import _add
from higher.optim import DifferentiableOptimizer
from higher.optim import _GroupedGradsType

import torch
from higher.patch import _MonkeyPatchBase
from torch import nn
from torch import optim
from torch import Tensor
from torch.optim.optimizer import required

from anatome.helper import LayerIdentifier, dist_data_set_per_layer

FuncModel = _MonkeyPatchBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NonDiffMAML(optim.Optimizer):  # copy pasted from torch.optim.SGD

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)


class MAML(DifferentiableOptimizer):  # copy pasted from DifferentiableSGD but with the g.detach() line of code

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        zipped = zip(self.param_groups, grouped_grads)
        for group_idx, (group, grads) in enumerate(zipped):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue

                if weight_decay != 0:
                    g = _add(g, weight_decay, p)
                if momentum != 0:
                    param_state = self.state[group_idx][p_idx]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = g
                    else:
                        buf = param_state['momentum_buffer']
                        buf = _add(buf.mul(momentum), 1 - dampening, g)
                        param_state['momentum_buffer'] = buf
                    if nesterov:
                        g = _add(g, momentum, buf)
                    else:
                        g = buf

                if self.fo:  # first-order
                    g = g.detach()  # dissallows flow of higher order grad while still letting params track gradients.
                group['params'][p_idx] = _add(p, -group['lr'], g)


higher.register_optim(NonDiffMAML, MAML)


def get_maml_inner_optimizer(model: nn.Module, inner_lr: float) -> NonDiffMAML:
    """
    This is meant to return the non-differentiable version and once you give it to the
    get_diff optimizer (or context loop), makes it differentiable. It's a higher detail.
    """
    # inner_opt = torch_uu.optim.SGD(self.base_model.parameters(), lr=self.inner_lr)
    inner_opt = NonDiffMAML(model.parameters(), lr=inner_lr)
    # inner_opt = torch_uu.optim.Adam(self.base_model.parameters(), lr=self.inner_lr)
    # self.args.inner_opt_name = str(inner_opt)
    return inner_opt


def get_diff_optimizer_and_functional_model(model: nn.Module,
                                            opt: optim.Optimizer,
                                            copy_initial_weights: bool,
                                            track_higher_grads: bool,
                                            override: Optional = None) \
        -> tuple[FuncModel, DifferentiableOptimizer]:
    """
    Creates a functional model (for higher) and differentiable optimizer (for higher).
    Replaces higher's context manager to return a differentiable optimizer and functional model:
            with higher.innerloop_ctx(base_model, inner_opt, copy_initial_weights=args.copy_initial_weights,
                                       track_higher_grads=args.track_higher_grads) as (fmodel, diffopt):

    ref:
        - https://github.com/facebookresearch/higher/blob/main/higher/__init__.py
        - https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
        - https://github.com/facebookresearch/higher/issues/119

    :param model:
    :param opt:
    :param copy_initial_weights: DONT PUT TRUE. details: set to True only if you do NOT want to train base model's
        initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    :param track_higher_grads: set to false during meta-testing but code sets it automatically only for meta-test
    :param override:
    :return:
    """
    from higher import monkeypatch
    from higher.patch import _MonkeyPatchBase
    from higher import optim
    from higher.optim import DifferentiableOptimizer
    # - Create a monkey-patched stateless version of a module.
    fmodel: _MonkeyPatchBase = monkeypatch(
        model,
        device,
        copy_initial_weights=copy_initial_weights,
        track_higher_grads=track_higher_grads
    )
    # - Construct/initialize a differentiable version of an existing optimizer.
    diffopt: DifferentiableOptimizer = optim.get_diff_optim(
        opt,
        model.parameters(),
        fmodel=fmodel,
        device=device,
        override=override,
        track_higher_grads=track_higher_grads
    )
    return fmodel, diffopt


def get_maml_adapted_model_with_higher_one_task(base_model: nn.Module,
                                       inner_opt: optim.Optimizer,
                                       spt_x_t: Tensor, spt_y_t: Tensor,
                                       training: bool,
                                       copy_initial_weights: bool,
                                       track_higher_grads: bool,
                                       fo: bool,
                                       nb_inner_train_steps: int,
                                       criterion: nn.Module) -> FuncModel:
    """
    Return an adaptated model using MAML using pytorch's higher lib.

    Decision of .eval() and .train():
        - when training we are doing base_model.trian() because that is what the official omniglot maml higher code is
        doing. Probably that is fine since during training even if the moel collects BN stats from different tasks, it's
        not a big deal (since it can only improve or worsen the performance but at least it does not "cheat" when reporting
        meta-test accuracy results).
        - whe meta-testing we always do .eval() to avoid task info jumping illegally from one place to another. When it
        solves a task (via spt, qry set) it only uses the BN stats from training (if it has them) or the current batch
        statistics (due to mdl.eval()).

    ref:
        - official higher maml omniglot: https://github.com/facebookresearch/higher/blob/main/examples/maml-omniglot.py
        - how to do this questioon on higher: https://github.com/facebookresearch/higher/issues/119
    """
    # - get fmodel and diffopt ready for inner adaptation
    # base_model.train() if training else base_model.eval()
    fmodel, diffopt = get_diff_optimizer_and_functional_model(base_model,
                                                              inner_opt,
                                                              copy_initial_weights=copy_initial_weights,
                                                              track_higher_grads=track_higher_grads)
    # - do inner addptation using task/support set
    print(f'>maml_new (before inner adapt): {fmodel.model.features.conv1.weight.norm(2)=}')
    diffopt.fo = fo
    for i_inner in range(nb_inner_train_steps):
        # base model forward pass
        spt_logits_t = fmodel(spt_x_t)
        inner_loss = criterion(spt_logits_t, spt_y_t)
        # inner-opt update
        diffopt.step(inner_loss)
    print(f'>maml_new (after inner adapt): {fmodel.model.features.conv1.weight.norm(2)=}')
    return fmodel


def _get_maml_adapted_model_with_higher_one_task_with_context_manager(base_model: nn.Module,
                                                             inner_opt: optim.Optimizer,
                                                             spt_x_t: Tensor, spt_y_t: Tensor,
                                                             training: bool,
                                                             copy_initial_weights: bool,
                                                             track_higher_grads: bool,
                                                             fo: bool,
                                                             nb_inner_train_steps: int,
                                                             criterion: nn.Module) -> FuncModel:
    """
    Return an adaptated model using MAML using pytorch's higher lib.

    Decision of .eval() and .train():
        - when training we are doing base_model.trian() because that is what the official omniglot maml higher code is
        doing. Probably that is fine since during training even if the moel collects BN stats from different tasks, it's
        not a big deal (since it can only improve or worsen the performance but at least it does not "cheat" when reporting
        meta-test accuracy results).
        - whe meta-testing we always do .eval() to avoid task info jumping illegally from one place to another. When it
        solves a task (via spt, qry set) it only uses the BN stats from training (if it has them) or the current batch
        statistics (due to mdl.eval()).

    ref:
        - official higher maml omniglot: https://github.com/facebookresearch/higher/blob/main/examples/maml-omniglot.py
        - how to do this questioon on higher: https://github.com/facebookresearch/higher/issues/119
    """
    # print('get_maml_adapted_model_with_higher_one_task_with_context_manager')
    # - get fmodel and diffopt ready for inner adaptation
    base_model.train() if training else base_model.eval()

    # - Inner Loop Adaptation
    with higher.innerloop_ctx(base_model, inner_opt, copy_initial_weights=copy_initial_weights,
                              track_higher_grads=track_higher_grads) as (fmodel, diffopt):
        # - do inner addptation using task/support set
        diffopt.fo = fo
        print(f'>maml_new (before inner adapt): {fmodel.model.features.conv1.weight.norm(2)=}')
        for i_inner in range(nb_inner_train_steps):
            # base model forward pass
            spt_logits_t = fmodel(spt_x_t)
            inner_loss = criterion(spt_logits_t, spt_y_t)
            # inner-opt update
            diffopt.step(inner_loss)
        print(f'>>maml_new (after inner adapt): {fmodel.model.features.conv1.weight.norm(2)=}')
        return fmodel


# def inner_loop():
#     meta_batch_size = spt_x.size(0)
#     meta_losses, meta_accs = [], []
#     for t in range(meta_batch_size):
#         spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]
#         # - Inner Loop Adaptation
#         with higher.innerloop_ctx(self.base_model, inner_opt, copy_initial_weights=self.args.copy_initial_weights,
#                                   track_higher_grads=self.args.track_higher_grads) as (fmodel, diffopt):
#             diffopt.fo = self.fo
#             for i_inner in range(self.args.nb_inner_train_steps):
#                 fmodel.train()
#
#                 # base/child model forward pass
#                 spt_logits_t = fmodel(spt_x_t)
#                 inner_loss = self.args.criterion(spt_logits_t, spt_y_t)
#                 # inner_train_err = calc_error(mdl=fmodel, X=S_x, Y=S_y)  # for more advanced learners like meta-lstm
#
#                 # inner-opt update
#                 diffopt.step(inner_loss)

# - comparing models with maml

def dist_batch_tasks_for_all_layer_mdl_vs_adapted_mdl(
        mdl: nn.Module,
        spt_x: Tensor, spt_y: Tensor, qry_x: Tensor, qry_y: Tensor,
        layer_names: list[str],
        inner_opt: DifferentiableOptimizer,
        fo: bool,
        nb_inner_train_steps: int,
        criterion: nn.Module,
        metric_comparison_type: str = 'pwcca',
        iters: int = 1,
        effective_neuron_type: str = 'filter',
        downsample_method: Optional[str] = None,
        downsample_size: Optional[int] = None,
        subsample_effective_num_data_method: Optional[str] = None,
        subsample_effective_num_data_param: Optional[int] = None,
        metric_as_sim_or_dist: str = 'dist',
        force_cpu: bool = False,
        training: bool = False,
        copy_initial_weights: bool = False,
        track_higher_grads: bool = False
) -> list[OrderedDict[LayerIdentifier, float]]:
    """
    :param mdl:
    :param spt_x: not as a tuple due to having to move them to gpu potentially.
    :param spt_y:
    :param qry_x:
    :param qry_y:
    :param layer_names:
    :param inner_opt:
    :param fo:
    :param nb_inner_train_steps:
    :param criterion:
    :param metric_comparison_type:
    :param iters:
    :param effective_neuron_type:
    :param downsample_method:
    :param downsample_size:
    :param subsample_effective_num_data_method:
    :param subsample_effective_num_data_param:
    :param metric_as_sim_or_dist:
    :param force_cpu:
    :param training:
    :param copy_initial_weights:
    :param track_higher_grads:
    :return:
    """
    # - [B, M, C, H, W] -> [B, L]
    L: int = len(layer_names)
    B: int = spt_x.size(0)
    dists_per_batch_per_layer: list[OrderedDict[LayerIdentifier, float]] = []
    for t in range(B):
        spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]
        #
        adapted_mdl: FuncModel = get_maml_adapted_model_with_higher_one_task(mdl,
                                                                    inner_opt,
                                                                    spt_x_t, spt_y_t,
                                                                    training,
                                                                    copy_initial_weights,
                                                                    track_higher_grads,
                                                                    fo,
                                                                    nb_inner_train_steps,
                                                                    criterion)
        # - [M, C, H, W], [L] -> [L]
        X: Tensor = qry_x_t
        dists_per_layer: OrderedDict[LayerIdentifier, float] = dist_data_set_per_layer(mdl1=mdl,
                                                                                       mdl2=adapted_mdl,
                                                                                       X1=X,
                                                                                       X2=X,
                                                                                       layer_names1=layer_names,
                                                                                       layer_names2=layer_names,
                                                                                       metric_comparison_type=metric_comparison_type,
                                                                                       iters=iters,
                                                                                       effective_neuron_type=effective_neuron_type,
                                                                                       downsample_method=downsample_method,
                                                                                       downsample_size=downsample_size,
                                                                                       subsample_effective_num_data_method=subsample_effective_num_data_method,
                                                                                       subsample_effective_num_data_param=subsample_effective_num_data_param,
                                                                                       metric_as_sim_or_dist=metric_as_sim_or_dist,
                                                                                       force_cpu=force_cpu
                                                                                       )
        assert len(dists_per_layer) == L
        # - appending to [B, L]
        dists_per_batch_per_layer.append(dists_per_layer)
        #
        # del adapted_mdl
        # gc.collect()
    assert len(dists_per_batch_per_layer) == B
    # Invariant due to asserts: [B, L] list

    # - [B, L] distances ready!
    return dists_per_batch_per_layer

# - meta-evaluation

def meta_eval_no_context_manager(args: Namespace, val_iterations: int = 0, save_val_ckpt: bool = True, split: str = 'val') -> tuple:
    """
    Evaluates the meta-learner on the given meta-set.

    ref for BN/eval:
        - https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
        - https://github.com/tristandeleu/pytorch-maml/issues/19
    """
    from uutils.torch_uu import process_meta_batch
    # - need to re-implement if you want to go through the entire data-set to compute an epoch (no more is ever needed)
    assert val_iterations == 0, f'Val iterations has to be zero but got {val_iterations}, if you want more precision increase (meta) batch size.'
    args.meta_learner.eval()
    for batch_idx, batch in enumerate(args.dataloaders[split]):
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)

        # Forward pass
        # eval_loss, eval_acc = args.meta_learner(spt_x, spt_y, qry_x, qry_y)
        # eval_loss, eval_acc = meta_learner_forward_adapt_batch_of_tasks(args.meta_learner, spt_x, spt_y, qry_x, qry_y, split=split)
        # eval_loss, eval_acc = forward(args.meta_learner, spt_x, spt_y, qry_x, qry_y)
        eval_loss, eval_acc = forward2(args.meta_learner, spt_x, spt_y, qry_x, qry_y)

        # store eval info
        if batch_idx >= val_iterations:
            break

    save_val_ckpt = False if split == 'test' else save_val_ckpt  # don't save models based on test set
    if float(eval_loss) < float(args.best_val_loss) and save_val_ckpt:
        args.best_val_loss = float(eval_loss)
        # from meta_learning.training.meta_training import save_for_meta_learning
        # save_for_meta_learning(args, ckpt_filename='ckpt_best_val.pt')
    return eval_loss, eval_acc


def meta_learner_forward_adapt_batch_of_tasks(meta_learner, spt_x, spt_y, qry_x, qry_y, split: str = 'train') -> tuple[float, float]:
    """
    Returns the acc & loss on the meta-batch of query sets.

    Note: crucially, this code uses the code that does not use the context manager from higher. This is so to
    test that code that is later use to compare models using ultimate-anatome.
    """
    training: bool = (split == 'train')
    # - get inner opt
    inner_opt = NonDiffMAML(meta_learner.base_model.parameters(), lr=meta_learner.lr_inner)
    print(f'{inner_opt=}')
    # inner_opt = get_maml_inner_optimizer(meta_learner.base_model, meta_learner.lr_inner)

    # - adapt
    # eval_loss, eval_acc = meta_learner(spt_x, spt_y, qry_x, qry_y)
    # return eval_loss, eval_acc
    meta_learner.base_model.train() if training else meta_learner.base_model.eval()
    meta_batch_size = spt_x.size(0)
    meta_losses, meta_accs = [], []
    for t in range(meta_batch_size):
        spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]
        # - Inner Loop Adaptation
        # fmodel: FuncModel = get_maml_adapted_model_with_higher_one_task(meta_learner.base_model,
        #                                                        inner_opt,
        #                                                        spt_x_t, spt_y_t,
        #                                                        training,
        #                                                        copy_initial_weights=meta_learner.args.copy_initial_weights,
        #                                                        track_higher_grads=meta_learner.args.track_higher_grads,
        #                                                        fo=meta_learner.fo,
        #                                                        nb_inner_train_steps=meta_learner.args.nb_inner_train_steps,
        #                                                        criterion=meta_learner.args.criterion)
        # fmodel: FuncModel = _get_maml_adapted_model_with_higher_one_task_with_context_manager(meta_learner.base_model,
        #                                                                 inner_opt,
        #                                                                 spt_x_t, spt_y_t,
        #                                                                 training,
        #                                                                 copy_initial_weights=meta_learner.args.copy_initial_weights,
        #                                                                 track_higher_grads=meta_learner.args.track_higher_grads,
        #                                                                 fo=meta_learner.fo,
        #                                                                 nb_inner_train_steps=meta_learner.args.nb_inner_train_steps,
        #                                                                 criterion=meta_learner.args.criterion)
        self = meta_learner
        with higher.innerloop_ctx(self.base_model, inner_opt, copy_initial_weights=self.args.copy_initial_weights,
                                  track_higher_grads=self.args.track_higher_grads) as (fmodel, diffopt):
            diffopt.fo = self.fo
            print(f'>maml_new (before inner adapt): {fmodel.model.features.conv1.weight.norm(2)=}')
            for i_inner in range(self.args.nb_inner_train_steps):
                # fmodel.train()  # omniglot doesn't have this here, it has a single one at the top https://github.com/facebookresearch/higher/blob/main/examples/maml-omniglot.py#L116

                # base/child model forward pass
                spt_logits_t = fmodel(spt_x_t)
                inner_loss = self.args.criterion(spt_logits_t, spt_y_t)
                # inner_train_err = calc_error(mdl=fmodel, X=S_x, Y=S_y)  # for more advanced learners like meta-lstm

                # inner-opt update
                diffopt.step(inner_loss)
            print(f'>>maml_new (after inner adapt): {fmodel.model.features.conv1.weight.norm(2)=}')

        # Evaluate on query set for current task
        qry_logits_t = fmodel(qry_x_t)
        qry_loss_t = meta_learner.args.criterion(qry_logits_t, qry_y_t)

        # Accumulate gradients wrt meta-params for each task: https://github.com/facebookresearch/higher/issues/104
        # note this is more mem efficient (removes intermediate data needed since backward has already been called)
        (qry_loss_t / meta_batch_size).backward()

        # get accuracy
        if meta_learner.target_type == 'classification':
            from uutils.torch_uu import calc_accuracy_from_logits
            qry_acc_t = calc_accuracy_from_logits(y_logits=qry_logits_t, y=qry_y_t)
        else:
            from uutils.torch_uu import r2_score_from_torch
            qry_acc_t = r2_score_from_torch(qry_y_t, qry_logits_t)

        # collect losses & accs
        meta_losses.append(qry_loss_t.item())
        meta_accs.append(qry_acc_t)
    assert len(meta_losses) == meta_batch_size
    meta_loss = np.mean(meta_losses)
    meta_acc = np.mean(meta_accs)
    return meta_loss, meta_acc


# --

# def _adapt_untested(meta_learner, spt_x, spt_y, qry_x, qry_y, split: str = 'train') -> tuple[float, float]:
#     """
#     """
#     # - get inner opt
#     inner_opt = NonDiffMAML(meta_learner.base_model.parameters(), lr=meta_learner.lr_inner)
#
#     # - adapt
#     meta_learner.base_model.train() if split == 'train' else meta_learner.base_model.eval()
#     meta_batch_size = spt_x.size(0)
#     meta_losses, meta_accs = [], []
#     for t in range(meta_batch_size):
#         spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]
#         # - Inner Loop Adaptation
#         with higher.innerloop_ctx(meta_learner.base_model, inner_opt,
#                                   copy_initial_weights=meta_learner.args.copy_initial_weights,
#                                   track_higher_grads=meta_learner.args.track_higher_grads) as (fmodel, diffopt):
#             diffopt.fo = meta_learner.fo
#             for i_inner in range(meta_learner.args.nb_inner_train_steps):
#                 fmodel.train()
#                 # base/child model forward pass
#                 spt_logits_t = fmodel(spt_x_t)
#                 inner_loss = meta_learner.args.criterion(spt_logits_t, spt_y_t)
#                 # inner-opt update
#                 diffopt.step(inner_loss)
#
#         # Evaluate on query set for current task
#         qry_logits_t = fmodel(qry_x_t)
#         qry_loss_t = meta_learner.args.criterion(qry_logits_t, qry_y_t)
#
#         # Accumulate gradients wrt meta-params for each task: https://github.com/facebookresearch/higher/issues/104
#         (
#                 qry_loss_t / meta_batch_size).backward()  # note this is more memory efficient (as it removes intermediate data that used to be needed since backward has already been called)
#
#         # get accuracy
#         if meta_learner.target_type == 'classification':
#             from uutils.torch_uu import calc_accuracy_from_logits
#             qry_acc_t = calc_accuracy_from_logits(y_logits=qry_logits_t, y=qry_y_t)
#         else:
#             from uutils.torch_uu import r2_score_from_torch
#             qry_acc_t = r2_score_from_torch(qry_y_t, qry_logits_t)
#
#         # collect losses & accs for logging/debugging
#         meta_losses.append(qry_loss_t.item())
#         meta_accs.append(qry_acc_t)
#
#     assert len(meta_losses) == meta_batch_size
#     meta_loss = np.mean(meta_losses)
#     meta_acc = np.mean(meta_accs)
#     return meta_loss, meta_acc

def forward(self, spt_x, spt_y, qry_x, qry_y):
    """Does L(A(theta,S), Q) = sum^N_{t=1} L(A(theta,S_t),Q_t) where A(theta,S) is the inner-adaptation loop.
    It also accumulates the gradient (for memory efficiency) for the outer-optimizer to later use

    Decision for BN/eval:
    - during training always use .train().
    During eval use the meta-train stats so do .eval() (and using .train() is always wrong since it cheats).
    Having track_running_stats=False seems overly complicated and nobody seems to use it...so why use it?

    ref for BN/eval:
        - https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
        - https://github.com/tristandeleu/pytorch-maml/issues/19

    Args:
        spt_x ([type]): x's for support set. Example shape [N,k_shot,D] D=1 or D=[C,H,W]
        spt_y ([type]): y's/target value for support set. Example shape [N,k_eval] or [N,k_eval,D]
        qry_x ([type]): x's for query set. Example shape [N,C,D] D=1 or D=[C,H,W]
        qry_y ([type]): y's/target value for query set. Example shape [N,k_eval] or [N,k_eval,D]

    Returns:
        [type]: [description]
    """
    # inner_opt = torch_uu.optim.SGD(self.base_model.parameters(), lr=self.lr_inner)
    inner_opt = NonDiffMAML(self.base_model.parameters(), lr=self.lr_inner)
    print(f'{inner_opt=}')
    # inner_opt = torch_uu.optim.Adam(self.base_model.parameters(), lr=self.lr_inner)
    self.args.inner_opt_name = str(inner_opt)

    self.base_model.train() if self.args.split == 'train' else self.base_model.eval()
    meta_batch_size = spt_x.size(0)
    meta_losses, meta_accs = [], []
    for t in range(meta_batch_size):
        spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]
        # - Inner Loop Adaptation
        with higher.innerloop_ctx(self.base_model, inner_opt, copy_initial_weights=self.args.copy_initial_weights,
                                  track_higher_grads=self.args.track_higher_grads) as (fmodel, diffopt):
            diffopt.fo = self.fo
            print(f'>maml_new (before inner adapt): {fmodel.model.features.conv1.weight.norm(2)=}')
            for i_inner in range(self.args.nb_inner_train_steps):
                # fmodel.train()  # omniglot doesn't have this here, it has a single one at the top https://github.com/facebookresearch/higher/blob/main/examples/maml-omniglot.py#L116

                # base/child model forward pass
                spt_logits_t = fmodel(spt_x_t)
                inner_loss = self.args.criterion(spt_logits_t, spt_y_t)
                # inner_train_err = calc_error(mdl=fmodel, X=S_x, Y=S_y)  # for more advanced learners like meta-lstm

                # inner-opt update
                diffopt.step(inner_loss)
        print(f'>>maml_new (after inner adapt): {fmodel.model.features.conv1.weight.norm(2)=}')

        # Evaluate on query set for current task
        qry_logits_t = fmodel(qry_x_t)
        qry_loss_t = self.args.criterion(qry_logits_t, qry_y_t)

        # Accumulate gradients wrt meta-params for each task: https://github.com/facebookresearch/higher/issues/104
        (qry_loss_t / meta_batch_size).backward()  # note this is more memory efficient (as it removes intermediate data that used to be needed since backward has already been called)

        # get accuracy
        if self.target_type == 'classification':
            from uutils.torch_uu import calc_accuracy_from_logits
            qry_acc_t = calc_accuracy_from_logits(y_logits=qry_logits_t, y=qry_y_t)
        else:
            from uutils.torch_uu import r2_score_from_torch
            qry_acc_t = r2_score_from_torch(qry_y_t, qry_logits_t)
            # qry_acc_t = compressed_r2_score(y_true=qry_y_t.detach().numpy(), y_pred=qry_logits_t.detach().numpy())

        # collect losses & accs for logging/debugging
        meta_losses.append(qry_loss_t.item())
        meta_accs.append(qry_acc_t)

    assert len(meta_losses) == meta_batch_size
    meta_loss = np.mean(meta_losses)
    meta_acc = np.mean(meta_accs)
    return meta_loss, meta_acc

def forward2(self, spt_x, spt_y, qry_x, qry_y):
    """
    """
    inner_opt = NonDiffMAML(self.base_model.parameters(), lr=self.lr_inner)
    self.args.inner_opt_name = str(inner_opt)

    self.base_model.train() if self.args.split == 'train' else self.base_model.eval()
    meta_batch_size = spt_x.size(0)
    meta_losses, meta_accs = [], []
    for t in range(meta_batch_size):
        spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]
        # - Inner Loop Adaptation
        # with higher.innerloop_ctx(self.base_model, inner_opt, copy_initial_weights=self.args.copy_initial_weights,
        #                           track_higher_grads=self.args.track_higher_grads) as (fmodel, diffopt):
        #     diffopt.fo = self.fo
        #     print(f'>maml_new (before inner adapt): {fmodel.model.features.conv1.weight.norm(2)=}')
        #     for i_inner in range(self.args.nb_inner_train_steps):
        #         # fmodel.train()  # omniglot doesn't have this here, it has a single one at the top https://github.com/facebookresearch/higher/blob/main/examples/maml-omniglot.py#L116
        #
        #         # base/child model forward pass
        #         spt_logits_t = fmodel(spt_x_t)
        #         inner_loss = self.args.criterion(spt_logits_t, spt_y_t)
        #         # inner_train_err = calc_error(mdl=fmodel, X=S_x, Y=S_y)  # for more advanced learners like meta-lstm
        #
        #         # inner-opt update
        #         diffopt.step(inner_loss)
        fmodel: FuncModel = get_maml_adapted_model_with_higher_one_task(self.base_model,
                                                               inner_opt,
                                                               spt_x_t, spt_y_t,
                                                               training=False,
                                                               copy_initial_weights=self.args.copy_initial_weights,
                                                               track_higher_grads=self.args.track_higher_grads,
                                                               fo=self.fo,
                                                               nb_inner_train_steps=self.args.nb_inner_train_steps,
                                                               criterion=self.args.criterion)
        # Evaluate on query set for current task
        qry_logits_t = fmodel(qry_x_t)
        qry_loss_t = self.args.criterion(qry_logits_t, qry_y_t)

        # Accumulate gradients wrt meta-params for each task: https://github.com/facebookresearch/higher/issues/104
        (qry_loss_t / meta_batch_size).backward()  # note this is more memory efficient (as it removes intermediate data that used to be needed since backward has already been called)

        # get accuracy
        if self.target_type == 'classification':
            from uutils.torch_uu import calc_accuracy_from_logits
            qry_acc_t = calc_accuracy_from_logits(y_logits=qry_logits_t, y=qry_y_t)
        else:
            from uutils.torch_uu import r2_score_from_torch
            qry_acc_t = r2_score_from_torch(qry_y_t, qry_logits_t)

        # collect losses & accs for logging/debugging
        meta_losses.append(qry_loss_t.item())
        meta_accs.append(qry_acc_t)

    assert len(meta_losses) == meta_batch_size
    meta_loss = np.mean(meta_losses)
    meta_acc = np.mean(meta_accs)
    return meta_loss, meta_acc