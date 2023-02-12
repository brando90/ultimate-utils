import logging
from argparse import Namespace
from typing import Callable

import learn2learn
import torch
import torch.nn as nn
from learn2learn.data import TaskDataset
from torch import Tensor
from torch.nn import functional as F
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

from uutils.torch_uu.metrics.confidence_intervals import torch_compute_confidence_interval

from pdb import set_trace as st

# - l2l


class GPTMetaLearnerL2L(nn.Module):
    def __init__(
            self,
            args,
            base_model,
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
        self.min_batch_size = min_batch_size

    def forward(self, batch, training: bool = True, call_backward: bool = False):
        """
        Does a forward pass ala l2l.

        Decision for BN/eval:
            - during meta-training always use .train(), see: https://stats.stackexchange.com/a/551153/28986
        """
        meta_batch_size: int = max(self.args.batch_size // self.args.world_size, 1)
        return self._forward(args=self.args, batch = batch, meta_batch_size=meta_batch_size, training=training,  # always true to avoid .eval()
            call_backward=call_backward,  # False for val/test
            )


    def eval_forward(self, batch, training: bool = True, call_backward: bool = False):
        meta_batch_size: int = max(self.args.batch_size // self.args.world_size, 1)
        return self._forward(args=self.args, batch = batch, meta_batch_size=meta_batch_size, training=training,  # always true to avoid .eval()
            call_backward=call_backward,  # False for val/test
            )


    def _forward(self, args: Namespace,
            batch,
            meta_batch_size: int,  # suggested max(batch_size or eval // args.world_size, 2)
            training: bool = True,  # always true to avoid .eval()
            call_backward: bool = False,  # not needed during testing/inference
            ):

        batch = (batch[0].to(args.device), batch[1].to(args.device))
        self.base_model.train() if training else self.base_model.eval()
        meta_losses, meta_accs = [], []
        # print('--start forward')
        for task in range(meta_batch_size):

            # -- Inner Loop Adaptation
            learner = self.maml.clone()
            loss, acc = self.fast_adapt(
                args=args,
                batch=batch,
                learner=learner,
                adaptation_steps=args.nb_inner_train_steps,
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

        return meta_loss, meta_loss_ci, meta_acc, meta_acc_ci


    def fast_adapt(self, args: Namespace,
               batch, learner, adaptation_steps, device) -> tuple[Tensor, Tensor]:
        """"""

        # Adapt the model
        for step in range(adaptation_steps):
            # - note the loss is usually in the final layer for my models
            adaptation_error, _ = self.loss_for_half(learner(batch[0]), batch[1], first_half = True)
            learner.adapt(adaptation_error)

        # Evaluate the adapted model
        eval_logits = learner(batch[0])
        evaluation_error, evaluation_accuracy = self.loss_for_half(eval_logits, batch[1], first_half = False)
        
        return evaluation_error, evaluation_accuracy

    def loss_for_half(self, logits, target, first_half, reduction = 'mean'):
        """
        Evaluates loss on either the first half or the second half of the batch
        """
        mid_point = self.base_model.config.block_size//2
        # print("logits.size:", logits.size())
        # print("target.size:", target.size())
        if first_half:
            cut_logits = logits[:, :mid_point, :]
            cut_target = target[:, :mid_point]
        else:
            cut_logits = logits[:, mid_point:, :]
            cut_target = target[:, mid_point:]

        # print("cut_logits.size:", cut_logits.size())
        # print("cut_target.size:", cut_target.size())

        # TODO: reshape can make it slower
        loss = F.cross_entropy(cut_logits.reshape(-1, cut_logits.size(-1)), cut_target.reshape(-1), ignore_index=-1, reduction = reduction)

        preds = torch.argmax(cut_logits, dim = 2)
        # print("preds.size:", preds.size())

        if reduction == 'none':
            # retain information for the batch to compute confidence intervals
            acc = torch.sum((preds == cut_target), dim = 1)/cut_target.shape[1]
        else:
            acc = torch.sum((preds == cut_target))/(cut_target.shape[0]*cut_target.shape[1])

        return loss, acc

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
    from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloader
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
    args.dataloaders: dict = get_meta_learning_dataloader(args)
    print(f'{args.dataloaders=}')
    assert args.data_option == 'torchmeta_miniimagenet', f'Err: {args.data_option=}'
    print()
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
    # check_torchmeta_4_tuple_works_with_meta_learner_agent()
    # check_training_loop_fit_one_batch()
    # check_training_meta_train_fixed_iterations()
    check_training_fo_maml()
    print('Done, all Tests Passed! \a')
