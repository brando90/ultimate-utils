from argparse import Namespace

def get_dataloader(args: Namespace) -> dict:
    args.data_set_path.exapenduser()
    data_set_path: str = str(args.data_set_path).lower()
    if 'mnist' in data_set_path:
        dataloaders: dict =
    elif 'cifar10' in data_set_path:
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid data set: got {data_set_path=}')
    args.dataloaders = dataloaders
    return dataloaders