from argparse import Namespace

import uutils
from uutils.torch_uu import AverageMeter
from uutils.torch_uu.agents import Agent
from uutils.torch_uu.distributed import print_dist


def main_train_loop_based_on_fixed_number_of_epochs(agent: Agent, args: Namespace):
    """
    Trains model one epoch at a time - i.e. it's epochs based rather than iteration based.

    :param args:
    :param start_epoch:
    :return:
    """
    print_dist('Starting training...')

    bar_epoch = uutils.get_good_progressbar(max_value=args.num_epochs)
    train_loss, train_acc = float('inf'), 0.0
    epochs: iter = range(start_epoch, start_epoch + args.num_epochs)
    assert len(epochs) == args.num_epochs
    for epoch_num in range(start_epoch, start_epoch + args.num_epochs):
        args.epoch_num = epoch_num
        # -- train for one epoch
        avg_loss = AverageMeter('train loss')
        avg_acc = AverageMeter('train accuracy')
        for i, batch in enumerate(self.dataloaders['train']):
            train_loss, train_acc = self.forward_one_batch(batch, training=True)
            avg_loss.update(train_loss.item(), self.args.batch_size)
            avg_acc.update(train_acc, self.args.batch_size)

            self.optimizer.zero_grad()
            train_loss.backward()  # each process synchronizes its gradients in the backward pass
            self.optimizer.step()  # the right update is done since all procs have the right synced grads

            # - scheduler, annealing/decaying the learning rate
            if (args.it % 500 == 0 and args.it != 0 and hasattr(args, 'scheduler')) or args.debug:
                args.scheduler.step() if (args.scheduler is not None) else None

        # -- log full epoch stats
        self.log_train_stats(self.args.epoch_n, avg_loss.item(), avg_acc.item(), save_val_ckpt=True)
        args.epoch_num += 1

    return avg_loss.item(), avg_acc.item()

def main_train_loop_until_convergence(agent, args: Namesspace, acc_tolerance: float = 1.0,
                                      train_loss_tolerance: float = 0.001):
    pass