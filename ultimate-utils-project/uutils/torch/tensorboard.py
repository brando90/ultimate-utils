from torch.utils.tensorboard import SummaryWriter  # https://deeplizard.com/learn/video/psexxmdrufm
from pathlib import Path

import numpy as np
# from torch.utils.tensorboard import SummaryWriter

def log_2_tb(args, tag1, tag2, it, loss, acc):
    tb = args.tb
    # tb = SummaryWriter(log_dir=args.current_logs_path)  # uncomment for documentation to work
    tb.add_scalar(tag1, loss, it)
    tb.add_scalar(tag2, acc, it)

def log_2_tb_metalearning(args, it, loss, acc_err, split):
    """
    :param acc_err:
    :param loss:
    :param it:
    :param args:
    :param split: train, val, test
    :return:
    """
    tb = args.tb
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
