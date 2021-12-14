from argparse import Namespace

import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Optimizer

import uutils
from uutils.torch_uu import r2_score_from_torch
from uutils.torch_uu.checkpointing_uu.meta_learning import save_for_meta_learning
from uutils.torch_uu.distributed import is_lead_worker


def log_sim_to_check_presence_of_feature_reuse(args: Namespace,
                                               it: int,

                                               spt_x, spt_y, qry_x, qry_y,  # these are multiple tasks

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
    # - is it epoch or iteration
    it_or_epoch: str = 'epoch_num' if args.training_mode == 'epochs' else 'it'
    sim_or_dist: str = 'sim'
    if hasattr(args, 'metrics_as_dist'):
        sim_or_dist: str = 'dist' if args.metrics_as_dist else sim_or_dist
    total_its: int = args.num_empochs if args.training_mode == 'epochs' else args.num_its

    if (it == total_its - 1 or force_log) and is_lead_worker(args.rank):
        # if (it % log_freq_for_detection_of_feature_reuse == 0 or it == total_its - 1 or force_log) and is_lead_worker(args.rank):
        if hasattr(args, 'metrics_as_dist'):
            sims = args.meta_learner.compute_functional_similarities(spt_x, spt_y, qry_x, qry_y, args.layer_names,
                                                                     parallel=parallel, iter_tasks=iter_tasks,
                                                                     metric_as_dist=args.metrics_as_dist)
        else:
            sims = args.meta_learner.compute_functional_similarities(spt_x, spt_y, qry_x, qry_y, args.layer_names,
                                                                     parallel=parallel, iter_tasks=iter_tasks)
        mean_layer_wise_sim, std_layer_wise_sim, mean_summarized_rep_sim, std_summarized_rep_sim = summarize_similarities(
            args, sims)

        # -- log (print)
        args.logger.log(f' \n------ {sim_or_dist} stats: {it_or_epoch}={it} ------')
        # - per layer
        # if show_layerwise_sims:
        print(f'---- Layer-Wise metrics ----')
        print(f'mean_layer_wise_{sim_or_dist} (per layer)')
        pprint(mean_layer_wise_sim)
        print(f'std_layer_wise_{sim_or_dist} (per layer)')
        pprint(std_layer_wise_sim)

        # - rep sim
        print(f'---- Representation metrics ----')
        print(f'mean_summarized_rep_{sim_or_dist} (summary for rep layer)')
        pprint(mean_summarized_rep_sim)
        print(f'std_summarized_rep_{sim_or_dist} (summary for rep layer)')
        pprint(std_summarized_rep_sim)
        args.logger.log(f' -- sim stats : {it_or_epoch}={it} --')

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


# - tests

def get_args() -> Namespace:
    args = uutils.parse_args_synth_agent()
    args = uutils.setup_args_for_experiment(args)
    return args

def valid_for_test(args: Namespace, mdl: nn.Module, save_val_ckpt: bool = False):
    import torch

    for t in range(1):
        x = torch.randn(args.batch_size, 5)
        y = (x**2 + x + 1).sum(dim=1)

        y_pred = mdl(x).squeeze(dim=1)
        val_loss, val_acc = mse_loss(y_pred, y), r2_score_from_torch(y_true=y, y_pred=y_pred)

    if val_loss.item() < args.best_val_loss and save_val_ckpt:
        args.best_val_loss = val_loss.item()
        # save_ckpt(args, args.mdl, args.optimizer, ckpt_name='ckpt_best_val.pt')
    return val_loss, val_acc

def train_for_test(args: Namespace, mdl: nn.Module, optimizer: Optimizer, scheduler = None):
    for it in range(50):
        x = torch.randn(args.batch_size, 5)
        y = (x**2 + x + 1).sum(dim=1)

        y_pred = mdl(x).squeeze(dim=1)
        train_loss, train_acc = mse_loss(y_pred, y), r2_score_from_torch(y_true=y, y_pred=y_pred)

        optimizer.zero_grad()
        train_loss.backward()  # each process synchronizes it's gradients in the backward pass
        optimizer.step()  # the right update is done since all procs have the right synced grads
        scheduler.step()

        if it % 2 == 0 and is_lead_worker(args.rank):
            log_train_val_stats(args, it, train_loss, train_acc, valid_for_test, save_val_ckpt=True, log_to_tb=True)
            if it % 10 == 0:
                # save_ckpt(args, args.mdl, args.optimizer)
                pass

    return train_loss, train_acc

def debug_test():
    args: Namespace = get_args()

    # - get mdl, opt, scheduler, etc
    from uutils.torch_uu.models import get_simple_model
    args.mdl = get_simple_model(in_features=5, hidden_features=20, out_features=1, num_layer=2)
    args.optimizer = torch.optim.Adam(args.mdl.parameters(), lr=1e-1)
    args.scheduler = torch.optim.lr_scheduler.ExponentialLR(args.optimizer, gamma=0.999, verbose=False)

    # - train
    train_loss, train_acc = train_for_test(args, args.mdl, args.optimizer, args.scheduler)
    print(f'{train_loss=}, {train_loss=}')

    # - eval
    val_loss, val_acc = valid_for_test(args, args.mdl)

    print(f'{val_loss=}, {val_acc=}')


if __name__ == '__main__':
    import time
    start = time.time()
    debug_test()
    duration_secs = time.time() - start
    print(f"Success, time passed: hours:{duration_secs / (60 ** 2)}, minutes={duration_secs / 60}, seconds={duration_secs}")
    print('Done!\a')