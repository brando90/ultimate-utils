#from torch.utils.tensorboard import SummaryWriter # https://deeplizard.com/learn/video/psexxmdrufm

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