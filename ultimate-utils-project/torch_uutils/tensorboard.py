from torch.utils.tensorboard import SummaryWriter  # https://deeplizard.com/learn/video/psexxmdrufm
from pathlib import Path

import numpy as np
# from torch.utils.tensorboard import SummaryWriter

def log_2_tb(args, tag1, tag2, epoch, loss, acc):
    tb = args.tb
    # tb = SummaryWriter(log_dir=args.current_logs_path)  # uncomment for documentation to work
    tb.add_scalar(tag1, loss, epoch)
    tb.add_scalar(tag2, acc, epoch)

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
