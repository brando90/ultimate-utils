# from torch_uu.utils.tensorboard import SummaryWriter  # https://deeplizard.com/learn/video/psexxmdrufm
from pathlib import Path

import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  # https://deeplizard.com/learn/video/psexxmdrufm


def log_2_tb(tb, args, it, tag1: str, loss: float, tag2: str, acc: float):
    # tb = SummaryWriter(log_dir=args.current_logs_path)  # uncomment for documentation to work
    # tag1 = tag1.replace(' ', '_')
    # tag2 = tag2.replace(' ', '_')
    tb.add_scalar(tag1, float(loss), it)
    tb.add_scalar(tag2, float(acc), it)

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
    if not hasattr(args, 'target_type'):
        tag1 = f'{split}_loss'
        tag2 = f'{split}_accuracy_R2'
    elif args.target_type == 'regression':
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
    if not hasattr(args, 'target_type'):
        tag1 = f'{split}_loss'
        tag2 = f'{split}_accuracy_R2'
    elif args.target_type == 'regression':
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
    from datetime import datetime
    print('\n---running tb test, writing ot a tb...')
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter  # https://deeplizard.com/learn/video/psexxmdrufm

    current_time: str = datetime.now().strftime('%b%d_%H-%M-%S')
    path: Path = Path(f'~/data/logs/logs_{current_time}/tb').expanduser()
    print(f'Test path: {path=}')
    # path.mkdir(parents=True, exist_ok=True)  # doesn't seem it's needed
    tb: SummaryWriter = SummaryWriter(log_dir=path)
    print(f'created {tb=}')

    for i in range(5):
        loss = i + np.random.normal(loc=0, scale=1)
        print(f'{i=}: {loss=}')
        tb.add_scalar('loss', loss, i)


if __name__ == '__main__':
    test()
    print('Done with tensorboard test!\a')
