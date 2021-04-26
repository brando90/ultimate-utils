# from torch.utils.tensorboard import SummaryWriter  # https://deeplizard.com/learn/video/psexxmdrufm
from pathlib import Path

import numpy as np
# from torch.utils.tensorboard import SummaryWriter

def log_2_tb(tb, args, it, tag1: str, loss, tag2: str, acc):
    # tb = SummaryWriter(log_dir=args.current_logs_path)  # uncomment for documentation to work
    # tag1 = tag1.replace(' ', '_')
    # tag2 = tag2.replace(' ', '_')
    tb.add_scalar(tag1, loss, it)
    tb.add_scalar(tag2, acc, it)

def log_2_tb_supervisedlearning(tb, args, it, loss, acc, split):
    """
    :param tb:
    :param acc:
    :param acc_err:
    :param loss:
    :param it:
    :param args:
    :param split: train, val, test
    :return:
    """
    if args.target_type == 'regression':
        tag1 = f'{split}_loss'
        tag2 = f'{split}_R2'
    elif args.target_type == 'classification':
        tag1 = f'{split}_loss'
        tag2 = f'{split}_accuracy'
    else:
        raise ValueError(f'Error: args.target_type = {args.target_type} not valid.')
    # tb = SummaryWriter(log_dir=args.current_logs_path)  # uncomment for documentation to work
    tb.add_scalar(tag1, loss, it)
    tb.add_scalar(tag2, acc, it)

def log_2_tb_metalearning(tb, args, it, loss, acc, split):
    """
    :param tb:
    :param acc:
    :param loss:
    :param it:
    :param args:
    :param split: train, val, test
    :return:
    """
    if args.target_type == 'regression':
        tag1 = f'meta-{split}_loss'
        tag2 = f'meta-{split}_R2'
    elif args.target_type == 'classification':
        tag1 = f'meta-{split}_loss'
        tag2 = f'meta-{split}_accuracy'
    else:
        raise ValueError(f'Error: args.target_type = {args.target_type} not valid.')
    # tb = SummaryWriter(log_dir=args.current_logs_path)  # uncomment for documentation to work
    tb.add_scalar(tag1, loss, it)
    tb.add_scalar(tag2, acc, it)

def log_2_tb_metalearning_old(tb, args, it, loss, acc_err, split):
    """
    :param acc:
    :param loss:
    :param it:
    :param args:
    :param split: train, val, test
    :return:
    """
    if args.target_type == 'regression':
        tag1 = f'meta-{split}_loss'
        tag2 = f'meta-{split}_regression_accuracy'
    elif args.target_type == 'classification':
        tag1 = f'meta-{split}_loss'
        tag2 = f'meta-{split}_accuracy'
    else:
        raise ValueError(f'Error: args.target_type = {args.target_type} not valid.')
    # tb = SummaryWriter(log_dir=args.current_logs_path)  # uncomment for documentation to work
    tb.add_scalar(tag1, loss, it)
    tb.add_scalar(tag2, acc_err, it)

# -- tests

def test():
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter  # https://deeplizard.com/learn/video/psexxmdrufm

    # path = Path('~/data/logs/tb/').expanduser()
    path = Path('~/logs/logs_Sep29_12-38-08_jobid_-1/tb').expanduser()
    tb = SummaryWriter(log_dir=path)
    # tb = SummaryWriter(log_dir=args.current_logs_path)

    for i in range(3):
        loss = i + np.random.normal(loc=0, scale=1)
        tb.add_scalar('loss', loss, i)


if __name__ == '__main__':
    test()
