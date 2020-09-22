#from torch.utils.tensorboard import SummaryWriter # https://deeplizard.com/learn/video/psexxmdrufm
from pathlib import Path

def log_train_to_tensorboard(args, **kwargs):
    """Logs to tensorboard given by experiment args.
    
    Arguments:
        args {Namespace} -- arguments for experiment
    """
    args.tb.add_scalar('Meta-Loss-train', kwargs['meta_loss'], kwargs['outer_i'])
    args.tb.add_scalar('Outer-error-train', kwargs['outer_train_acc'], kwargs['outer_i'])

def log_val_to_tensorboard(args, **kwargs):
    """Logs to tensorboard given by experiment args.
    
    Arguments:
        args {Namespace} -- arguments for experiment
    """
    args.tb.add_scalar('Meta-Loss-val', kwargs['loss_mean'], kwargs['outer_i'])
    args.tb.add_scalar('Outer-error-val', kwargs['acc_mean'], kwargs['outer_i'])

def test():
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter  # https://deeplizard.com/learn/video/psexxmdrufm

    path = Path('~/data/logs/').expanduser()
    tb = SummaryWriter(log_dir=path)
    # tb = SummaryWriter(log_dir=args.current_logs_path)

    for i in range(3):
        loss = i + np.random.normal(loc=0, scale=1)
        tb.add_scalar('loss', loss, i)

if __name__ == '__main__':
    test()