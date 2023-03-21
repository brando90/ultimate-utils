import logging
from argparse import Namespace
from typing import Callable, Any

import torch
import torch.nn as nn
from torch import Tensor
# from torch.multiprocessing import Pool
# from torch.optim.optimizer import required
# from torch.optim import Optimizer as Optimizer
#
# import higher
# from higher.optim import _add
# from higher.optim import DifferentiableOptimizer
# from higher.optim import _GroupedGradsType

import uutils
from uutils.torch_uu import functional_diff_norm, ned_torch, r2_score_from_torch, calc_accuracy_from_logits, \
    normalize_matrix_for_similarity, process_meta_batch
from uutils.torch_uu import tensorify

import numpy as np

from uutils.torch_uu.meta_learners.maml_differentiable_optimizer import \
    get_lists_losses_accs_meta_learner_forward_adapt_batch_of_tasks
from uutils.torch_uu.metrics.confidence_intervals import torch_compute_confidence_interval

from pdb import set_trace as st

from copy import deepcopy


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
        self.nb_inner_train_steps = deepcopy(self.args.nb_inner_train_steps)
        self.inner_lr = deepcopy(self.args.inner_lr)
        if not hasattr(args, 'fo'):
            self.args.fo = True
        self.fo = deepcopy(self.args.fo)

        self.target_type = target_type

        self.inner_debug = inner_debug

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
        return meta_loss, meta_acc

    def eval_forward(self, batch, training: bool = True, call_backward: bool = False):
        """
        Does a forward pass ala l2l. It's the same as forward just so that all Agents have the same interface.
        This one looks redundant and it is, but it's here for consistency with the SL agents.
        The eval forward is different in SL agents.
        """
        loss, loss_ci, acc, acc_ci = eval_forward(self, batch, training=training, call_backward=call_backward)
        return loss, loss_ci, acc, acc_ci

    def get_lists_accs_losses(self, batch, training: bool = True, call_backward: bool = False):
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(self.args, batch)
        # note bellow code is already in forward, but we need to call it here to get the lists explicitly (no redundant code! ;) )
        meta_losses, meta_accs = get_lists_losses_accs_meta_learner_forward_adapt_batch_of_tasks(self, spt_x, spt_y,
                                                                                                 qry_x, qry_y,
                                                                                                 training,
                                                                                                 call_backward)
        return meta_losses, meta_accs

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
    import learn2learn
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
        # st()
        learner.adapt(adaptation_error)
        # st()

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


def get_lists_accs_losses_l2l(meta_learner,
                              args: Namespace,
                              task_dataset,  # from learn2learn.data import TaskDataset,
                              # args.tasksets.train, args.tasksets.validation or args.tasksets.test
                              meta_batch_size: int,  # suggested max(batch_size or eval // args.world_size, 2)

                              training: bool = True,  # always true to avoid .eval()
                              call_backward: bool = False,  # not needed during testing/inference
                              ):
    """
    Returns the (meta) accs & losses on the meta-batch/task_dataset.

    Note:
        - training true ensures .eval() is never called (due to BN, we always want batch stats)
        - call_backward collects gradients for outer_opt. Due to optimization of calling it here, we have the option to call it or not.
    """
    print(f'{task_dataset=}')
    assert args is meta_learner.args
    # -
    from learn2learn.data import TaskDataset
    task_dataset: TaskDataset = task_dataset  # args.tasksets.train, args.tasksets.validation or args.tasksets.test

    # - adapt
    meta_learner.base_model.train() if training else meta_learner.base_model.eval()
    meta_losses, meta_accs = [], []
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
            adaptation_steps=meta_learner.nb_inner_train_steps,
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
    return meta_losses, meta_accs


class MAMLMetaLearnerL2L(nn.Module):
    def __init__(
            self,
            args,
            base_model,

            target_type='classification',
    ):
        import learn2learn
        super().__init__()
        self.args = args  # args for experiment
        assert args is self.args
        self.base_model = base_model
        assert base_model is args.model
        self.inner_lr = deepcopy(args.inner_lr)
        self.nb_inner_train_steps = deepcopy(args.nb_inner_train_steps)
        self.first_order = deepcopy(args.first_order)
        # learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?
        allow_unused = args.allow_unused if hasattr(args, 'allow_unused') else None
        self.maml = learn2learn.algorithms.MAML(self.base_model,
                                                lr=self.inner_lr,
                                                first_order=self.first_order,
                                                allow_unused=allow_unused
                                                )
        # maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
        # opt = torch.optim.Adam(maml.parameters(), meta_lr)
        # opt = cherry.optim.Distributed(maml.parameters(), opt=opt, sync=1)

        self.target_type = target_type

    def forward(self, task_dataset, training: bool = True, call_backward: bool = False):
        """
        Does a forward pass ala l2l.

        Decision for BN/eval:
            - during meta-training always use .train(), see: https://stats.stackexchange.com/a/551153/28986
        """
        # - type task_dataset (since we don't want to globally import learn2learn here if you're not using it but still needs stuff in this file)
        from learn2learn.data import TaskDataset
        task_dataset: TaskDataset = task_dataset  # args.tasksets.train, args.tasksets.validation or args.tasksets.test
        # assert self.args.batch_size_eval == task_dataset.num_tasks, f"Err: {self.args.batch_size_eva} != {task_dataset.num_tasks}"
        # assert self.args.batch_size == task_dataset.num_tasks, f"Err: {self.args.batch_size} != {task_dataset.num_tasks}"
        # meta_batch_size: int = max(self.args.batch_size // self.args.world_size, 1)
        # meta_batch_size: int = max(task_dataset.num_tasks // self.args.world_size, 1)
        # meta_losses, meta_accs = get_lists_accs_losses_l2l(self, self.args, task_dataset, meta_batch_size, training,
        #                                                    call_backward)
        meta_losses, meta_accs = self.get_lists_accs_losses(task_dataset, training, call_backward)
        loss, loss_ci = torch_compute_confidence_interval(tensorify(meta_losses))
        acc, acc_ci = torch_compute_confidence_interval(tensorify(meta_accs))
        return loss, acc

    def eval_forward(self, task_dataset, training: bool = True, call_backward: bool = False):
        """
        Does a forward pass ala l2l. It's the same as forward just so that all Agents have the same interface.
        This one looks redundant and it is, but it's here for consistency with the SL agents, since
        the eval forward is different in SL agents (e.g. torch.no_grad is used in SL agent but here we don't).
        """
        # - type task_dataset (since we don't want to globally import learn2learn here if you're not using it but still needs stuff in this file)
        from learn2learn.data import TaskDataset
        task_dataset: TaskDataset = task_dataset  # args.tasksets.train, args.tasksets.validation or args.tasksets.test
        # assert self.args.batch_size_eval == task_dataset.num_tasks, f"Err: {self.args.batch_size_eva} != {task_dataset.num_tasks}"
        # assert self.args.batch_size == task_dataset.num_tasks, f"Err: {self.args.batch_size} != {task_dataset.num_tasks}"
        # meta_batch_size: int = max(self.args.batch_size // self.args.world_size, 1)
        # meta_batch_size: int = max(task_dataset.num_tasks // self.args.world_size, 1)
        # meta_losses, meta_accs = get_lists_accs_losses_l2l(self, self.args, task_dataset, meta_batch_size, training,
        #                                                    call_backward)
        loss, loss_ci, acc, acc_ci = eval_forward(self, task_dataset, training=training, call_backward=call_backward)
        return loss, loss_ci, acc, acc_ci

    def get_lists_accs_losses(self, task_dataset, training: bool = True, call_backward: bool = False):
        """
        Returns the acc & loss on the meta-batch from the task in task_dataset.
        """
        # -
        from learn2learn.data import TaskDataset
        task_dataset: TaskDataset = task_dataset  # args.tasksets.train, args.tasksets.validation or args.tasksets.test
        # -
        # meta_batch_size: int = max(self.args.batch_size // self.args.world_size, 1)
        meta_batch_size: int = max(task_dataset.num_tasks // self.args.world_size, 1)
        meta_batch_size: int = max(self.args.batch_size_eval // self.args.world_size, 1)
        # note bellow code is already in forward, but we need to call it here to get the lists explicitly (no redundant code! ;) )
        meta_losses, meta_accs = get_lists_accs_losses_l2l(meta_learner=self,
                                                           args=self.args,
                                                           task_dataset=task_dataset,  # eg args.tasksets.train
                                                           training=training,  # always true to avoid .eval()
                                                           meta_batch_size=meta_batch_size,
                                                           call_backward=call_backward,  # False for val/test
                                                           )
        return meta_losses, meta_accs

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


# -- eval code

def eval_forward(model: nn.Module, data: Any, training: bool = False, call_backward: bool = False):
    """
    Note:
        - training = True makes sense for meta-learning (or if you want norms to use batch statistics, but it might
        change your running statistics).
    """
    from uutils.torch_uu.metrics.confidence_intervals import mean_confidence_interval
    assert call_backward == False, 'call_backward should be False for eval_forward'
    losses, accs = model.get_lists_accs_losses(data, training, call_backward=call_backward)
    loss, loss_ci = mean_confidence_interval(losses)
    acc, acc_ci = mean_confidence_interval(accs)
    return loss, loss_ci, acc, acc_ci


# --

def get_minimum_args_to_run_maml_torchmeta_on_mi_5cnn() -> Namespace:
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_defaul_args_for_5cnn
    from pathlib import Path
    from uutils.argparse_uu.meta_learning import parse_args_meta_learning
    from uutils.argparse_uu.common import setup_args_for_experiment
    args: Namespace = parse_args_meta_learning()
    args = get_defaul_args_for_5cnn(args)
    args.data_option = 'torchmeta_miniimagenet'
    args.data_path = Path('~/data/torchmeta_data/').expanduser()
    args.inner_lr = 0.1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    # args.track_higher_grads = True  # not FO maml
    args.track_higher_grads = False  # fo? https://github.com/facebookresearch/higher/issues/63
    args.training_mode = 'iterations'
    args.num_its = 2
    args: Namespace = setup_args_for_experiment(args)
    return args


# - tests

# def check_training_loop_fit_one_batch():
#     from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
#     from uutils.torch_uu.training.meta_training import meta_train_agent_fit_single_batch
#     from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloader
#     from uutils.torch_uu.optim_uu.adam_uu import get_opt_adam_default
#     from uutils.torch_uu.optim_uu.adam_uu import get_cosine_scheduler_adam_rfs_cifarfs
#
#     args = get_minimum_args_to_run_maml_torchmeta_on_mi_5cnn()
#     args.training_mode = 'meta_train_agent_fit_single_batch'
#     args.model, args.model_hps = get_default_learner_and_hps_dict()
#     args.agent = MAMLMetaLearner(args, base_model=args.model)
#     args.meta_learner = args.agent
#     opt_hps = {}
#     args.opt, args.opt_hps = get_opt_adam_default(args.model, **opt_hps)
#     scheduler_hps = {}
#     args.scheduler, args.scheduler_hps = get_cosine_scheduler_adam_rfs_cifarfs(args.opt, **scheduler_hps)
#     args.dataloaders: dict = get_meta_learning_dataloader(args)
#     print(f'{args.dataloaders=}')
#     assert args.data_option == 'torchmeta_miniimagenet', f'Err: {args.data_option=}'
#     meta_train_agent_fit_single_batch(args, args.agent, args.dataloaders, args.opt, args.scheduler)

def check_training_fo_maml():
    print('---- checking fo MAML torchmeta higher')
    track_higher_grads = False
    print(f'{track_higher_grads=}')
    check_training_meta_train_fixed_iterations(track_higher_grads)
    print('---- success! Of fo MAML torchmeta higher')


def check_training_meta_train_fixed_iterations(track_higher_grads: bool = True):
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
    from uutils.torch_uu.training.meta_training import meta_train_fixed_iterations
    from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloaders
    from uutils.torch_uu.optim_uu.adam_uu import get_opt_adam_default
    from uutils.torch_uu.optim_uu.adam_uu import get_cosine_scheduler_adam_rfs_cifarfs

    args = get_minimum_args_to_run_maml_torchmeta_on_mi_5cnn()
    args.track_higher_grads = track_higher_grads
    args.model, args.model_hps = get_default_learner_and_hps_dict()
    args.agent = MAMLMetaLearner(args, base_model=args.model)
    args.meta_learner = args.agent
    opt_hps = {}
    args.opt, args.opt_hps = get_opt_adam_default(args.model, **opt_hps)
    scheduler_hps = {}
    args.scheduler, args.scheduler_hps = get_cosine_scheduler_adam_rfs_cifarfs(args.opt, **scheduler_hps)
    args.dataloaders: dict = get_meta_learning_dataloaders(args)
    print(f'{args.dataloaders=}')
    assert args.data_option == 'torchmeta_miniimagenet', f'Err: {args.data_option=}'
    print()
    meta_train_fixed_iterations(args, args.agent, args.dataloaders, args.opt, args.scheduler)


def check_torchmeta_4_tuple_works_with_meta_learner_agent():
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_defaul_args_for_5cnn
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_learner_from_args
    from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloaders
    from pathlib import Path
    from uutils.argparse_uu.meta_learning import parse_args_meta_learning
    from uutils.argparse_uu.common import setup_args_for_experiment

    args: Namespace = parse_args_meta_learning()
    args = get_defaul_args_for_5cnn(args)
    args.data_option = 'torchmeta_miniimagenet'
    args.data_path = Path('~/data/torchmeta_data/').expanduser()
    args.inner_lr = 0.1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = False  # fo? https://github.com/facebookresearch/higher/issues/63
    args: Namespace = setup_args_for_experiment(args)
    model = get_learner_from_args(args)  # random 5cnn
    agent = MAMLMetaLearner(args, base_model=model)

    args.dataloaders: dict = get_meta_learning_dataloaders(args)
    print(f'{args.dataloaders=}')
    assert args.data_option == 'torchmeta_miniimagenet', f'Err: {args.data_option=}'
    for batch in args.dataloaders['train']:
        losses = agent(batch)
        print(f'{losses=}')
        break


if __name__ == "__main__":
    # check_torchmeta_4_tuple_works_with_meta_learner_agent()
    # check_training_loop_fit_one_batch()
    # check_training_meta_train_fixed_iterations()
    check_training_fo_maml()
    print('Done, all Tests Passed! \a')
