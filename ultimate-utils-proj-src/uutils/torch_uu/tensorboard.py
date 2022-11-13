# from torch_uu.utils.tensorboard import SummaryWriter  # https://deeplizard.com/learn/video/psexxmdrufm
from pathlib import Path


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


def tensorboard_run_list_2_matplotlib_list(data: list[tuple], smoothing_weight: float) -> tuple[list, list]:
    """
    :param data: data in format [..., [time, it, value], ...]
        e.g [[1603380383.1535034, 200, 1.5554816722869873], [1603381593.4793968, 900, 1.235633373260498]
    :return:
    """
    its: list[int] = []
    values: list[float] = []
    for _, it, value in data:
        its.append(it)
        values.append(value)
    values = my_tb_smooth(scalars=values, weight=smoothing_weight)
    return its, values


def my_tb_smooth(scalars: list[float], weight: float) -> list[float]:  # Weight between 0 and 1
    """

    ref: https://stackoverflow.com/questions/42011419/is-it-possible-to-call-tensorboard-smooth-function-manually

    :param scalars:
    :param weight:
    :return:
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed: list = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed


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
