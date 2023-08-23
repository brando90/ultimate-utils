import logging
from argparse import Namespace
from typing import Any, Optional

from torch import Tensor, tensor, nn

from uutils.torch_uu.agents.common import Agent

# -- get loss & acc
from uutils.torch_uu.training.common import get_data


def do_eval(args: Namespace,
            model: Agent,
            dataloaders,
            split: str = 'val',
            training: bool = True,  # True to avoid different tasks: https://stats.stackexchange.com/a/551153/28986
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    note:
        - note we have "get_eval" function cuz the api for data loaders for meta-learning is different from the SL one.
        Despite the Agent having the same api as the MetaLearner object.
        - assumption: your agent has the .forward interface needed
    """
    # - get loss & acc
    if not hasattr(model, 'eval_forward'):  # for SL Agents since they tend to use torch.no_grad, maml both behave same.
        from uutils.torch_uu.metrics.confidence_intervals import mean_confidence_interval
        data: Any = get_data(dataloaders, split)
        losses, accs = model.get_lists_accs_losses(data, training)
        loss, loss_ci = mean_confidence_interval(losses)
        acc, acc_ci = mean_confidence_interval(accs)
    else:
        data: Any = get_data(dataloaders, split)
        loss, loss_ci, acc, acc_ci = model.eval_forward(data, training)
    return loss, loss_ci, acc, acc_ci

def eval_sl_gpt2_half_loss(args: Namespace,
            model: Agent,
            dataloaders,
            split: str = 'val',
            training: bool = False,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Evaluate the current model on the eval set (val set strongly recommended), but compute loss only on second half
    """
    batch: Any = next(iter(dataloaders[split]))
    # batch: Any = next(iter(dataloaders['test']))
    val_loss, val_loss_ci, val_acc, val_acc_ci = model.eval_forward(batch, training, half_loss = True)
    return val_loss, val_loss_ci, val_acc, val_acc_ci


# - tests, tutorials, examples

def eval_test_():
    # - usl
    from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
    from uutils.argparse_uu.supervised_learning import get_args_mi_usl_default
    from uutils.torch_uu.agents.supervised_learning import ClassificationSLAgent
    args: Namespace = get_args_mi_usl_default()
    get_and_create_model_opt_scheduler_for_run(args)
    args.agent = ClassificationSLAgent(args, args.model)
    from uutils.torch_uu.dataloaders.helpers import get_sl_dataloader
    args.dataloaders = get_sl_dataloader(args)
    loss, loss_ci, acc, acc_ci = do_eval(args, args.agent, args.dataloaders)
    print(f'{loss, loss_ci, acc, acc_ci=}')
    # - torchmeta
    from uutils.argparse_uu.meta_learning import get_args_mi_torchmeta_default
    from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
    from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner
    args: Namespace = get_args_mi_torchmeta_default()
    get_and_create_model_opt_scheduler_for_run(args)
    args.agent = MAMLMetaLearner(args, args.model)
    from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloaders
    args.dataloaders = get_meta_learning_dataloaders(args)
    loss, loss_ci, acc, acc_ci = do_eval(args, args.agent, args.dataloaders)
    print(f'{loss, loss_ci, acc, acc_ci=}')
    # - l2l
    from uutils.argparse_uu.meta_learning import get_args_mi_l2l_default
    from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
    from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
    from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearnerL2L
    args: Namespace = get_args_mi_l2l_default()
    get_and_create_model_opt_scheduler_for_run(args)
    args.agent = MAMLMetaLearnerL2L(args, args.model)
    args.dataloaders = get_l2l_tasksets(args)
    loss, loss_ci, acc, acc_ci = do_eval(args, args.agent, args.dataloaders)
    print(f'{loss, loss_ci, acc, acc_ci=}')


# - run main

if __name__ == '__main__':
    eval_test_()
