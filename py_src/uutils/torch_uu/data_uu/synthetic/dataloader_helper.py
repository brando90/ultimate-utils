import torch.nn as nn

from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import ClassSplitter
from meta_learning.datasets.rand_fnn import RandFNN

from meta_learning.training.meta_training import process_meta_batch

def get_randfnn_dataloader(args):
    args.criterion = nn.MSELoss()
    # get data
    print(args.data_path)
    dataset_train = RandFNN(args.data_path, 'train')
    dataset_val = RandFNN(args.data_path, 'val')
    dataset_test = RandFNN(args.data_path, 'test')
    # get meta-sets
    metaset_train = ClassSplitter(dataset_train,
                                  num_train_per_class=args.k_shots,
                                  num_test_per_class=args.k_eval,
                                  shuffle=True)
    metaset_val = ClassSplitter(dataset_val, num_train_per_class=args.k_shots,
                                num_test_per_class=args.k_eval,
                                shuffle=True)
    metaset_test = ClassSplitter(dataset_test, num_train_per_class=args.k_shots,
                                 num_test_per_class=args.k_eval,
                                 shuffle=True)
    # get meta-dataloader
    meta_train_dataloader = BatchMetaDataLoader(metaset_train,
                                                batch_size=args.meta_batch_size_train,
                                                num_workers=args.num_workers)
    meta_val_dataloader = BatchMetaDataLoader(metaset_val,
                                              batch_size=args.meta_batch_size_eval,
                                              num_workers=args.num_workers)
    meta_test_dataloader = BatchMetaDataLoader(metaset_test,
                                               batch_size=args.meta_batch_size_eval,
                                               num_workers=args.num_workers)

    return meta_train_dataloader, meta_val_dataloader, meta_test_dataloader

def get_sine_dataloader(args):
    from torchmeta.toy.helpers import sinusoid

    args.criterion = nn.MSELoss()
    # tran = transforms.Compose([torch_uu.tensor])
    dataset = sinusoid(shots=args.k_shots, test_shots=args.k_eval)
    meta_train_dataloader = BatchMetaDataLoader(dataset, batch_size=args.meta_batch_size_train,
                                                num_workers=args.num_workers)
    meta_val_dataloader = BatchMetaDataLoader(dataset, batch_size=args.meta_batch_size_eval,
                                              num_workers=args.num_workers)
    meta_test_dataloader = BatchMetaDataLoader(dataset, batch_size=args.meta_batch_size_eval,
                                               num_workers=args.num_workers)
    return meta_train_dataloader, meta_val_dataloader, meta_test_dataloader


def test():
    from types import SimpleNamespace

    args = SimpleNamespace(k_shots=10, k_eval=15, meta_batch_size_train=2, meta_batch_size_eval=2, num_workers=2)
    meta_train_dataloader, meta_val_dataloader, meta_test_dataloade = get_sine_dataloader(args)
    print(meta_val_dataloader)
    for idx, batch in enumerate(meta_val_dataloader):
        print(f'idx = {idx}')
        print(type(batch))
        if idx >= 10:
            return meta_val_dataloader

def test2():
    import time
    from tqdm import tqdm
    from types import SimpleNamespace

    args = SimpleNamespace(k_shots=10, k_eval=15, meta_batch_size_train=2, meta_batch_size_eval=2, num_workers=0)
    args.iters = 5
    meta_train_dataloader, meta_val_dataloader, meta_test_dataloade = get_sine_dataloader(args)
    print(meta_val_dataloader)
    print(len(meta_val_dataloader))
    with tqdm(range(args.iters)) as pbar:
        it = 0
        while it < args.iters:
            for batch_idx, batch in enumerate(meta_val_dataloader):
                print(f'\nit = {it}')
                spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
                print(spt_x.size())
                print(type(batch))
                time.sleep(0.5)
                print('done work for current iteration')
                it += 1
                pbar.update()
                if it >= args.iters:
                    break


if __name__ == '__main__':
    test2()
    print('Done!\a')
