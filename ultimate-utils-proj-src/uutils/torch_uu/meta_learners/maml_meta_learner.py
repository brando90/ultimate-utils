import logging
from argparse import Namespace
from typing import Callable

import learn2learn
import torch
import torch.nn as nn
from learn2learn.data import TaskDataset
from torch import Tensor
from torch.multiprocessing import Pool
from torch.optim.optimizer import required
from torch.optim import Optimizer as Optimizer

import higher
from higher.optim import _add
from higher.optim import DifferentiableOptimizer
from higher.optim import _GroupedGradsType

import uutils
from uutils.torch_uu import functional_diff_norm, ned_torch, r2_score_from_torch, calc_accuracy_from_logits, \
    normalize_matrix_for_similarity, process_meta_batch
from uutils.torch_uu import tensorify

import numpy as np

from uutils.torch_uu.metrics.confidence_intervals import torch_compute_confidence_interval

from pdb import set_trace as st

Spt_x, Spt_y, Qry_x, Qry_y = torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
Task = tuple[Spt_x, Spt_y, Qry_x, Qry_y]
Batch = list


class EmptyOpt(Optimizer):  # This is just an example
    def __init__(self, params, *args, **kwargs):
        defaults = {'args': args, 'kwargs': kwargs}
        super().__init__(params, defaults)


class NonDiffMAML(Optimizer):  # copy pasted from torch.optim.SGD

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

                # if self.fo:  # first-order
                #     g = g.detach()  # dissallows flow of higher order grad while still letting params track gradients.
                group['params'][p_idx] = _add(p, -group['lr'], g)


higher.register_optim(NonDiffMAML, MAML)


class MAMLMetaLearner(nn.Module):
    def __init__(
            self,
            args,
            base_model,

            inner_debug=False,
            target_type='classification'
    ):
        super().__init__()
        self.args = args  # args for experiment
        self.base_model = base_model
        # assert base_model is args.model

        self.inner_debug = inner_debug
        self.target_type = target_type

    @property
    def lr_inner(self) -> float:
        if hasattr(self.args, 'inner_lr'):
            return self.args.inner_lr
        else:
            return self.args.lr_inner

    @lr_inner.setter
    def lr_inner(self, new_val: float):
        if hasattr(self.args, 'inner_lr'):
            self.args.inner_lr = new_val
        else:
            self.args.lr_inner = new_val
        # self.args.inner_lr = new_val

    @property
    def fo(self) -> bool:
        if hasattr(self.args, 'fo'):
            if hasattr(self.args, 'track_higher_grads'):
                assert self.args.fo == (not self.args.track_higher_grads), f'Err: ' \
                                                                           f'{self.args.self.fo=} {self.args.track_higher_grads=}'
        if hasattr(self.args, 'fo'):
            return self.args.fo
        elif hasattr(self.args, 'track_higher_grads'):
            return not self.args.track_higher_grads
        else:
            raise ValueError(f'Err: {self.args.self.fo=} {self.args.track_higher_grads=}')

    # @property
    # def mdl(self) -> torch.nn.Module:
    #     return uutils.torch.get_model(self.mdl_)

    def forward(self, batch, training: bool = True, call_backward: bool = False):
        """
        Does L(A(theta,S), Q) = sum^N_{t=1} L(A(theta,S_t),Q_t) where A(theta,S) is the inner-adaptation loop.
        It also accumulates the gradient (for memory efficiency) for the outer-optimizer to later use

        Decision for BN/eval:
            - during meta-training always use .train(), see: https://stats.stackexchange.com/a/551153/28986
        """
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(self.args, batch)
        from uutils.torch_uu.meta_learners.maml_differentiable_optimizer import \
            meta_learner_forward_adapt_batch_of_tasks
        meta_loss, meta_loss_ci, meta_acc, meta_acc_ci = meta_learner_forward_adapt_batch_of_tasks(self, spt_x, spt_y,
                                                                                                   qry_x, qry_y,
                                                                                                   training,
                                                                                                   call_backward)
        return meta_loss, meta_loss_ci, meta_acc, meta_acc_ci

    def eval_forward(self, batch, training: bool = True, call_backward: bool = False):
        meta_loss, meta_loss_ci, meta_acc, meta_acc_ci = self.forward(batch, training, call_backward)
        return meta_loss, meta_loss_ci, meta_acc, meta_acc_ci

    def eval(self):
        """
        Note: decision is to do .train() for all meta-train
        ref: https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
        """
        logging.warning('Calling MAML.eval(). You sure you want to do that?')
        raise ValueError(
            f'Why are you calling eval during meta-learning? Read: https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du')
        self.base_model.eval()

    def parameters(self):
        return self.base_model.parameters()

    def regression(self):
        self.target_type = 'regression'
        self.args.target_type = 'regression'

    def classification(self):
        self.target_type = 'classification'
        self.args.target_type = 'classification'

    def cuda(self):
        self.base_model.cuda()


# - l2l

def fast_adapt(args: Namespace,
               task_data, learner, loss, adaptation_steps, shots, ways, device) -> tuple[Tensor, Tensor]:
    """"""
    # [n*(k+k_eval), C, H, W] (or [n(k+k_eval), D])
    data, labels = task_data
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    # [n*(k+k_eval), C, H, W] -> [n*k, C, H, W] and [n*k_eval, C, H, W]
    (support_data, support_labels), (query_data, query_labels) = learn2learn.data.partition_task(
        data=data,
        labels=labels,
        shots=shots,  # shots to separate to two data sets of size shots and k_eval
    )
    # checks coordinate 0 of size() [n*(k + k_eval), C, H, W]
    assert support_data.size(0) == shots * ways, f' Expected {shots * ways} but got {support_data.size(0)}'
    # checks [n*k] since these are the labels
    assert support_labels.size() == torch.Size([shots * ways])

    # Adapt the model
    for step in range(adaptation_steps):
        # - note the loss is usually in the final layer for my models
        adaptation_error = loss(learner(support_data), support_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions: Tensor = learner(query_data)
    evaluation_error: Tensor = loss(predictions, query_labels)

    # get accuracy
    if args.agent.target_type == 'classification':
        evaluation_accuracy: Tensor = learn2learn.utils.accuracy(predictions, query_labels)
    else:
        from uutils.torch_uu import r2_score_from_torch
        evaluation_accuracy = r2_score_from_torch(query_labels, predictions)
        raise NotImplementedError
    return evaluation_error, evaluation_accuracy


def forward(meta_learner,
            args: Namespace,
            task_dataset: TaskDataset,  # args.tasksets.train, args.tasksets.validation or args.tasksets.test
            meta_batch_size: int,  # suggested max(batch_size or eval // args.world_size, 2)

            training: bool = True,  # always true to avoid .eval()
            call_backward: bool = False,  # not needed during testing/inference
            ):
    """
    Returns the acc & loss on the meta-batch from the task in task_dataset.

    Note:
    - training true ensures .eval() is never called (due to BN, we always want batch stats)
    - call_backward collects gradients for outer_opt. Due to optimization of calling it here, we have the option
    to call it or not.
    """
    assert args is meta_learner.args
    assert args.meta_learner is meta_learner
    assert args.agent is meta_learner

    # - adapt
    meta_learner.base_model.train() if training else meta_learner.base_model.eval()
    meta_losses, meta_accs = [], []
    # print('--start forward')
    for task in range(meta_batch_size):
        # print(f'{task=}')
        # - Sample all data data for spt & qry sets for current task: thus size [n*(k+k_eval), C, H, W] (or [n(k+k_eval), D])
        task_data: list = task_dataset.sample()  # data, labels

        # -- Inner Loop Adaptation
        learner = meta_learner.maml.clone()
        loss, acc = fast_adapt(
            args=args,
            task_data=task_data,
            learner=learner,
            loss=args.loss,
            adaptation_steps=args.nb_inner_train_steps,
            shots=args.k_shots,
            ways=args.n_classes,
            device=args.device,
        )
        if call_backward:
            loss.backward()
        # collect losses & accs
        meta_losses.append(loss.item())
        meta_accs.append(acc.item())
    assert len(meta_losses) == meta_batch_size
    assert len(meta_accs) == meta_batch_size
    meta_loss, meta_loss_ci = torch_compute_confidence_interval(tensorify(meta_losses))
    meta_acc, meta_acc_ci = torch_compute_confidence_interval(tensorify(meta_accs))
    # print('-- done forward --')
    return meta_loss, meta_loss_ci, meta_acc, meta_acc_ci


class MAMLMetaLearnerL2L(nn.Module):
    def __init__(
            self,
            args,
            base_model,

            target_type='classification',
            min_batch_size=1,
    ):
        super().__init__()
        self.args = args  # args for experiment
        self.base_model = base_model
        assert args is self.args
        assert base_model is args.model
        allow_unused = args.allow_unused if hasattr(args, 'allow_unused') else None  # ternary op for backwards comp.
        self.maml = learn2learn.algorithms.MAML(args.model,
                                                lr=args.inner_lr,
                                                first_order=args.first_order,
                                                allow_unused=allow_unused
                                                )
        # maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
        # opt = torch.optim.Adam(maml.parameters(), meta_lr)
        # opt = cherry.optim.Distributed(maml.parameters(), opt=opt, sync=1)

        self.target_type = target_type
        self.min_batch_size = min_batch_size

    def forward(self, task_dataset: TaskDataset, training: bool = True, call_backward: bool = False):
        """
        Does a forward pass ala l2l.

        Decision for BN/eval:
            - during meta-training always use .train(), see: https://stats.stackexchange.com/a/551153/28986
        """
        meta_batch_size: int = max(self.args.batch_size // self.args.world_size, 1)
        meta_loss, meta_loss_ci, meta_acc, meta_acc_ci = forward(meta_learner=self,
                                                                 args=self.args,
                                                                 task_dataset=task_dataset,  # eg args.tasksets.train
                                                                 training=training,  # always true to avoid .eval()
                                                                 meta_batch_size=meta_batch_size,

                                                                 call_backward=call_backward,  # False for val/test
                                                                 )
        return meta_loss, meta_loss_ci, meta_acc, meta_acc_ci

    def eval_forward(self, task_dataset: TaskDataset, training: bool = True, call_backward: bool = False):
        meta_batch_size: int = max(self.args.batch_size // self.args.world_size, 1)
        meta_loss, meta_loss_ci, meta_acc, meta_acc_ci = forward(meta_learner=self,
                                                                 args=self.args,
                                                                 task_dataset=task_dataset,  # eg args.tasksets.train
                                                                 training=training,  # always true to avoid .eval()
                                                                 meta_batch_size=meta_batch_size,

                                                                 call_backward=call_backward,  # False for val/test
                                                                 )
        return meta_loss, meta_loss_ci, meta_acc, meta_acc_ci

    def eval(self):
        """
        Note: decision is to do .train() for all meta-train and .eval() for meta-eval.
        ref: https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
        """
        logging.warning('Calling MAML.eval(). You sure you want to do that?')
        self.base_model.eval()

    def parameters(self):
        # return self.base_model.parameters()
        # todo would be nice to check if self.maml.parameters() and self.base_model.parameters() are the same
        return self.maml.parameters()

    def regression(self):
        self.target_type = 'regression'
        self.args.target_type = 'regression'

    def classification(self):
        self.target_type = 'classification'
        self.args.target_type = 'classification'

    def cuda(self):
        self.base_model.cuda()


def get_minimum_args_to_run_maml_torchmeta_on_mi_5cnn() -> Namespace:
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_defaul_args_for_5cnn
    from pathlib import Path
    from uutils.argparse_uu.meta_learning import parse_args_meta_learning
    from uutils.argparse_uu.common import setup_args_for_experiment
    args: Namespace = parse_args_meta_learning()
    args = get_defaul_args_for_5cnn(args)
    args.data_option = 'torchmeta_miniimagenet'
    args.data_path = Path('~/data/torchmeta_data/').expanduser()
    args.lr_inner = 0.1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # not FO maml
    # args.track_higher_grads = False  # fo? https://github.com/facebookresearch/higher/issues/63
    args: Namespace = setup_args_for_experiment(args)
    return args


# - tests

def check_training_loop_fit_one_batch():
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
    from uutils.torch_uu.training.meta_training import meta_train_agent_fit_single_batch
    from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloader
    from uutils.torch_uu.optim_uu.adam_uu import get_opt_adam_default
    from uutils.torch_uu.optim_uu.adam_uu import get_cosine_scheduler_adam_rfs_cifarfs

    args = get_minimum_args_to_run_maml_torchmeta_on_mi_5cnn()
    args.training_mode = 'meta_train_agent_fit_single_batch'
    args.model, args.model_hps = get_default_learner_and_hps_dict()
    args.agent = MAMLMetaLearner(args, base_model=args.model)
    args.meta_learner = args.agent
    opt_hps = {}
    args.opt, args.opt_hps = get_opt_adam_default(args.model, **opt_hps)
    scheduler_hps = {}
    args.scheduler, args.scheduler_hps = get_cosine_scheduler_adam_rfs_cifarfs(args.opt, **scheduler_hps)
    args.dataloaders: dict = get_meta_learning_dataloader(args)
    print(f'{args.dataloaders=}')
    assert args.data_option == 'torchmeta_miniimagenet', f'Err: {args.data_option=}'
    meta_train_agent_fit_single_batch(args, args.agent, args.dataloaders, args.opt, args.scheduler)


def check_training_meta_train_fixed_iterations():
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
    from uutils.torch_uu.training.meta_training import meta_train_fixed_iterations
    from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloader
    from uutils.torch_uu.optim_uu.adam_uu import get_opt_adam_default
    from uutils.torch_uu.optim_uu.adam_uu import get_cosine_scheduler_adam_rfs_cifarfs

    args = get_minimum_args_to_run_maml_torchmeta_on_mi_5cnn()
    args.training_mode = 'iterations'
    args.num_its = 2
    args.model, args.model_hps = get_default_learner_and_hps_dict()
    args.agent = MAMLMetaLearner(args, base_model=args.model)
    args.meta_learner = args.agent
    opt_hps = {}
    args.opt, args.opt_hps = get_opt_adam_default(args.model, **opt_hps)
    scheduler_hps = {}
    args.scheduler, args.scheduler_hps = get_cosine_scheduler_adam_rfs_cifarfs(args.opt, **scheduler_hps)
    args.dataloaders: dict = get_meta_learning_dataloader(args)
    print(f'{args.dataloaders=}')
    assert args.data_option == 'torchmeta_miniimagenet', f'Err: {args.data_option=}'
    meta_train_fixed_iterations(args, args.agent, args.dataloaders, args.opt, args.scheduler)


def check_torchmeta_4_tuple_works_with_meta_learner_agent():
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_defaul_args_for_5cnn
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_learner_from_args
    from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloader
    from pathlib import Path
    from uutils.argparse_uu.meta_learning import parse_args_meta_learning
    from uutils.argparse_uu.common import setup_args_for_experiment

    args: Namespace = parse_args_meta_learning()
    args = get_defaul_args_for_5cnn(args)
    args.data_option = 'torchmeta_miniimagenet'
    args.data_path = Path('~/data/torchmeta_data/').expanduser()
    args.lr_inner = 0.1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    # args.track_higher_grads = True  # not FO maml
    args.track_higher_grads = False  # fo? https://github.com/facebookresearch/higher/issues/63
    args: Namespace = setup_args_for_experiment(args)
    model = get_learner_from_args(args)  # random 5cnn
    agent = MAMLMetaLearner(args, base_model=model)

    args.dataloaders: dict = get_meta_learning_dataloader(args)
    print(f'{args.dataloaders=}')
    assert args.data_option == 'torchmeta_miniimagenet', f'Err: {args.data_option=}'
    for batch in args.dataloaders['train']:
        losses = agent(batch)
        print(f'{losses=}')
        break


if __name__ == "__main__":
    check_torchmeta_4_tuple_works_with_meta_learner_agent()
    # check_training_loop_fit_one_batch()
    check_training_meta_train_fixed_iterations()
    print('Done, all Tests Passed! \a')
