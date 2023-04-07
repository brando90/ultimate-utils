import os
from argparse import Namespace

from torch.utils.data import Dataset

from uutils.torch_uu.dataset.concate_dataset import ConcatDatasetMutuallyExclusiveLabels

# - FC100 (fungi too long), cu_b, dtd

def fc100_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='fc100',
        device=None,
) -> dict:
    print(f'----> {data_augmentation=}')
    print(f'{fc100_usl_all_splits_dataloaders=}')
    root = os.path.expanduser(root)
    from diversity_src.dataloaders.maml_patricks_l2l import get_fc100_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_fc100_list_data_set_splits(root,
                                                                                                         data_augmentation,
                                                                                                         device)
    # - print the number of classes in each split
    # print('-- Printing num classes')
    # from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    # get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # - concat l2l datasets to get usl single dataset
    relabel_filename: str = 'fc100_train_relabel_usl.pt'
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train, root, relabel_filename)
    relabel_filename: str = 'fc100_val_relabel_usl.pt'
    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation, root, relabel_filename)
    relabel_filename: str = 'fc100_test_relabel_usl.pt'
    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test, root, relabel_filename)
    assert len(train_dataset.labels) == 60, f'Err:\n{len(train_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders

def cu_birds_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='cu_birds',
        device=None,
) -> dict:
    print(f'----> {data_augmentation=}')
    print(f'{cu_birds_usl_all_splits_dataloaders=}')
    root = os.path.expanduser(root)
    from diversity_src.dataloaders.maml_patricks_l2l import get_cu_birds_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_cu_birds_list_data_set_splits(root,
                                                                                                         data_augmentation,
                                                                                                         device)
    # - print the number of classes in each split
    # print('-- Printing num classes')
    # from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    # get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # - concat l2l datasets to get usl single dataset
    relabel_filename: str = 'cu_birds_train_relabel_usl.pt'
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train, root, relabel_filename)
    relabel_filename: str = 'cu_birds_val_relabel_usl.pt'
    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation, root, relabel_filename)
    relabel_filename: str = 'cu_birds_test_relabel_usl.pt'
    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test, root, relabel_filename)
    assert len(train_dataset.labels) == 140, f'Err:\n{len(train_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders


def dtd_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='dtd',
        device=None,
) -> dict:
    print(f'----> {data_augmentation=}')
    print(f'{dtd_usl_all_splits_dataloaders=}')
    root = os.path.expanduser(root)
    from diversity_src.dataloaders.maml_patricks_l2l import get_dtd_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_dtd_list_data_set_splits(root,
                                                                                                         data_augmentation,
                                                                                                         device)
    # - print the number of classes in each split
    # print('-- Printing num classes')
    # from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    # get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # - concat l2l datasets to get usl single dataset
    relabel_filename: str = 'dtd_train_relabel_usl.pt'
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train, root, relabel_filename)
    relabel_filename: str = 'dtd_val_relabel_usl.pt'
    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation, root, relabel_filename)
    relabel_filename: str = 'dtd_test_relabel_usl.pt'
    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test, root, relabel_filename)
    assert len(train_dataset.labels) == 33, f'Err:\n{len(train_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders


def aircraft_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb5_vggair',
        device=None,
) -> dict:
    print(f'----> {data_augmentation=}')
    print(f'{aircraft_usl_all_splits_dataloaders=}')
    root = os.path.expanduser(root)
    from diversity_src.dataloaders.maml_patricks_l2l import get_aircraft_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_aircraft_list_data_set_splits(root,
                                                                                                         data_augmentation,
                                                                                                         device)
    # - print the number of classes in each split
    # print('-- Printing num classes')
    # from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    # get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # - concat l2l datasets to get usl single dataset
    relabel_filename: str = 'aircraft_train_relabel_usl.pt'
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train, root, relabel_filename)
    relabel_filename: str = 'aircraft_val_relabel_usl.pt'
    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation, root, relabel_filename)
    relabel_filename: str = 'aircraft_test_relabel_usl.pt'
    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test, root, relabel_filename)
    #assert len(train_dataset.labels) == 33, f'Err:\n{len(train_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders


def flower_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb5_vggair',
        device=None,
) -> dict:
    print(f'----> {data_augmentation=}')
    print(f'{flower_usl_all_splits_dataloaders=}')
    root = os.path.expanduser(root)
    from diversity_src.dataloaders.maml_patricks_l2l import get_flower_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_flower_list_data_set_splits(root,
                                                                                                         data_augmentation,
                                                                                                         device)
    # - print the number of classes in each split
    # print('-- Printing num classes')
    # from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    # get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # - concat l2l datasets to get usl single dataset
    relabel_filename: str = 'flower_train_relabel_usl.pt'
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train, root, relabel_filename)
    relabel_filename: str = 'flower_val_relabel_usl.pt'
    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation, root, relabel_filename)
    relabel_filename: str = 'flower_test_relabel_usl.pt'
    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test, root, relabel_filename)
    #assert len(train_dataset.labels) == 33, f'Err:\n{len(train_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders



def fungi_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='fungi',
        device=None,
) -> dict:
    print(f'----> {data_augmentation=}')
    print(f'{fungi_usl_all_splits_dataloaders=}')
    root = os.path.expanduser(root)
    from diversity_src.dataloaders.maml_patricks_l2l import get_fungi_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_fungi_list_data_set_splits(root,
                                                                                                         data_augmentation,
                                                                                                         device)
    # - print the number of classes in each split
    # print('-- Printing num classes')
    # from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    # get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # - concat l2l datasets to get usl single dataset
    relabel_filename: str = 'fungi_train_relabel_usl.pt'
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train, root, relabel_filename)
    relabel_filename: str = 'fungi_val_relabel_usl.pt'
    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation, root, relabel_filename)
    relabel_filename: str = 'fungi_test_relabel_usl.pt'
    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test, root, relabel_filename)
    #assert len(train_dataset.labels) == 33, f'Err:\n{len(train_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders

def delaunay_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb4_micod',
        device=None,
) -> dict:
    print(f'----> {data_augmentation=}')
    print(f'{delaunay_usl_all_splits_dataloaders=}')
    root = os.path.expanduser(root)
    from diversity_src.dataloaders.maml_patricks_l2l import get_delaunay_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_delaunay_list_data_set_splits(root,
                                                                                                         data_augmentation,
                                                                                                         device)
    # - print the number of classes in each split
    # print('-- Printing num classes')
    # from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    # get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # - concat l2l datasets to get usl single dataset
    relabel_filename: str = 'delaunay_train_relabel_usl.pt'
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train, root, relabel_filename)
    relabel_filename: str = 'delaunay_val_relabel_usl.pt'
    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation, root, relabel_filename)
    relabel_filename: str = 'delaunay_test_relabel_usl.pt'
    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test, root, relabel_filename)
    assert len(train_dataset.labels) == 34, f'Err:\n{len(train_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders



def hdb6_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb4_micod',
        device=None,
) -> dict:
    print(f'----> {data_augmentation=}')
    print(f'{delaunay_usl_all_splits_dataloaders=}')
    root = os.path.expanduser(root)
    from diversity_src.dataloaders.maml_patricks_l2l import get_hdb6_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_hdb6_list_data_set_splits(root,
                                                                                                         data_augmentation,
                                                                                                         device)
    # - print the number of classes in each split
    # print('-- Printing num classes')
    # from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    # get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # - concat l2l datasets to get usl single dataset
    relabel_filename: str = 'hdb6_train_relabel_usl.pt'
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train, root, relabel_filename)
    relabel_filename: str = 'hdb6_val_relabel_usl.pt'
    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation, root, relabel_filename)
    relabel_filename: str = 'hdb6_test_relabel_usl.pt'
    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test, root, relabel_filename)
    #assert len(train_dataset.labels) == 34, f'Err:\n{len(train_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders




def hdb7_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb4_micod',
        device=None,
) -> dict:
    print(f'----> {data_augmentation=}')
    print(f'{delaunay_usl_all_splits_dataloaders=}')
    root = os.path.expanduser(root)
    from diversity_src.dataloaders.maml_patricks_l2l import get_hdb7_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_hdb7_list_data_set_splits(root,
                                                                                                         data_augmentation,
                                                                                                         device)
    # - print the number of classes in each split
    # print('-- Printing num classes')
    # from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    # get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # - concat l2l datasets to get usl single dataset
    relabel_filename: str = 'hdb7_train_relabel_usl.pt'
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train, root, relabel_filename)
    relabel_filename: str = 'hdb7_val_relabel_usl.pt'
    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation, root, relabel_filename)
    relabel_filename: str = 'hdb7_test_relabel_usl.pt'
    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test, root, relabel_filename)
    #assert len(train_dataset.labels) == 34, f'Err:\n{len(train_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders


def hdb8_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb4_micod',
        device=None,
) -> dict:
    print(f'----> {data_augmentation=}')
    #print(f'{delaunay_usl_all_splits_dataloaders=}')
    root = os.path.expanduser(root)
    from diversity_src.dataloaders.maml_patricks_l2l import get_hdb8_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_hdb8_list_data_set_splits(root,
                                                                                                         data_augmentation,
                                                                                                         device)
    # - print the number of classes in each split
    # print('-- Printing num classes')
    # from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    # get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # - concat l2l datasets to get usl single dataset
    relabel_filename: str = 'hdb8_train_relabel_usl.pt'
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train, root, relabel_filename)
    relabel_filename: str = 'hdb8_val_relabel_usl.pt'
    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation, root, relabel_filename)
    relabel_filename: str = 'hdb8_test_relabel_usl.pt'
    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test, root, relabel_filename)
    #assert len(train_dataset.labels) == 34, f'Err:\n{len(train_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders


def hdb9_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb4_micod',
        device=None,
) -> dict:
    print(f'----> {data_augmentation=}')
    #print(f'{delaunay_usl_all_splits_dataloaders=}')
    root = os.path.expanduser(root)
    from diversity_src.dataloaders.maml_patricks_l2l import get_hdb9_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_hdb9_list_data_set_splits(root,
                                                                                                         data_augmentation,
                                                                                                         device)
    # - print the number of classes in each split
    # print('-- Printing num classes')
    # from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    # get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # - concat l2l datasets to get usl single dataset
    relabel_filename: str = 'hdb9_train_relabel_usl.pt'
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train, root, relabel_filename)
    relabel_filename: str = 'hdb9_val_relabel_usl.pt'
    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation, root, relabel_filename)
    relabel_filename: str = 'hdb9_test_relabel_usl.pt'
    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test, root, relabel_filename)
    #assert len(train_dataset.labels) == 34, f'Err:\n{len(train_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders


def hdb10_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb4_micod',
        device=None,
) -> dict:
    print(f'----> {data_augmentation=}')
    #print(f'{delaunay_usl_all_splits_dataloaders=}')
    root = os.path.expanduser(root)
    from diversity_src.dataloaders.maml_patricks_l2l import get_hdb10_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_hdb10_list_data_set_splits(root,
                                                                                                         data_augmentation,
                                                                                                         device)
    # - print the number of classes in each split
    # print('-- Printing num classes')
    # from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    # get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # - concat l2l datasets to get usl single dataset
    relabel_filename: str = 'hdb10_train_relabel_usl.pt'
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train, root, relabel_filename)
    relabel_filename: str = 'hdb10_val_relabel_usl.pt'
    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation, root, relabel_filename)
    relabel_filename: str = 'hdb10_test_relabel_usl.pt'
    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test, root, relabel_filename)
    #assert len(train_dataset.labels) == 34, f'Err:\n{len(train_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders
# - gen labels
def get_len_labels_list_datasets(datasets: list[Dataset], verbose: bool = False) -> int:
    if verbose:
        print('--- get_len_labels_list_datasets')
        print([len(dataset.labels) for dataset in datasets])
        print([dataset.labels for dataset in datasets])
        print('--- get_len_labels_list_datasets')
    return sum([len(dataset.labels) for dataset in datasets])









# - tests

def loop_through_usl_hdb_and_pass_data_through_mdl():
    print(f'starting {loop_through_usl_hdb_and_pass_data_through_mdl=} test')
    # - for determinism
    import random
    random.seed(0)
    import torch
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)

    # - args
    args = Namespace(batch_size=8, batch_size_eval=2, rank=-1, world_size=1)

    # - get data loaders
    # dataloaders: dict = hdb1_mi_omniglot_usl_all_splits_dataloaders(args)
    #dataloaders: dict = dtd_usl_all_splits_dataloaders(args)
    #dataloaders = fungi_usl_all_splits_dataloaders(args)

    for dataloaders in [hdb8_usl_all_splits_dataloaders(args),hdb9_usl_all_splits_dataloaders(args),hdb10_usl_all_splits_dataloaders(args)]: #dtd_usl_all_splits_dataloaders(args), cu_birds_usl_all_splits_dataloaders(args), fc100_usl_all_splits_dataloaders(args),
        print(dataloaders['train'].dataset.labels)
        print(dataloaders['val'].dataset.labels)
        print(dataloaders['test'].dataset.labels)
        n_train_cls: int = len(dataloaders['train'].dataset.labels)
        print('-- got the usl hdb data loaders --')

        # - loop through tasks
        import torch
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        # model = get_model('resnet18', pretrained=False, num_classes=n_train_cls).to(device)
        # model = get_model('resnet18', pretrained=True, num_classes=n_train_cls).to(device)
        from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_mi
        model, _ = get_resnet_rfs_model_mi('resnet12_rfs', num_classes=n_train_cls)
        model.to(device)
        from torch import nn
        criterion = nn.CrossEntropyLoss()
        for split, dataloader in dataloaders.items():
            print(f'-- {split=}')
            # next(iter(dataloaders[split]))
            for it, batch in enumerate(dataloaders[split]):
                print(f'{it=}')

                X, y = batch
                print(f'{X.size()=}')
                print(f'{y.size()=}')
                print(f'{y=}')

                y_pred = model(X)
                loss = criterion(y_pred, y)
                print(f'{loss=}')
                print()
                break

    print('-- end of test --')


if __name__ == '__main__':
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    loop_through_usl_hdb_and_pass_data_through_mdl()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
