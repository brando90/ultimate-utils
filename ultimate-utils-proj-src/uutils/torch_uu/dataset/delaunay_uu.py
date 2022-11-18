"""
Some comments about the original Delauny data set:
The final dataset comprises 11, 503 samples across 53
classes, i.e., artists (mean number of samples per artist: 217.04;
standard deviation: 58.55), along with their source URLs.

These samples are split between a training set of 9202 images, and a test set of 2301 images.

Due to the heterogeneous nature
of sources, images vary significantly in their resolution, from
80px × 80px for the smallest sample to 3365px × 4299px for
the largest.

ref:
- https://arxiv.org/abs/2201.12123
- data set to learn2learn task set: https://github.com/learnables/learn2learn/issues/375

Comments on previous experiments:
I'm trying to see how the diversity of a data set affects the meta-learning. For the conclusions to be consistent/sound in my experiments the experimental settings have to be similar for all experiments. Meaning:
- 1. since I used train models with data augmentation (hypothesized to make more diverse data sets) evaluate models on test without data augmentation & computed/evaluated the diversity on data sets without data augmentation, we should keep that consistent in future experiments.
    - diversities for train can be computed out of curiosity but should not be used for conclusion unless previous evaluations are repeated, in particular, train diversities have to be computed.

- Hypothesis: data set diversity is affect by data augmentation. For now all training is done with data augmentation since that is what I did in training (consistent/sound conclusions). Also, data augmentation is done in practice so it helps our experiments be more representative of what is done in practice.
    - will compute train diversities out of curiosity, put in appendix, but unlikely to do much with them unless I recompute them for previous experiments.

We use data augementation so our conclusions apply to what is done in practice but use the data augmentation strategy
that is most consistent as possible (to not have it be a source of diversity and instead having the data itself be the
main source).

def get_imagenet_data_transform():
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    #
    # val_dataset = datasets.ImageFolder(
    #     valdir,
    #     transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    pass
"""
import torch
from datetime import datetime

from argparse import Namespace

import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, Compose, ToPILImage, RandomCrop, ColorJitter, RandomHorizontalFlip, \
    ToTensor, RandomResizedCrop, Resize, Normalize, Pad

from typing import Union

from pathlib import Path

from uutils import download_and_extract, expanduser, copy_folders_recursively

mean = [0.5853, 0.5335, 0.4950]
std = [0.2348, 0.2260, 0.2242]

classes = ('Ad Reinhardt', 'Alberto Magnelli', 'Alfred Manessier', 'Anthony Caro',
           'Antoine Pevsner', 'Auguste Herbin', 'Aurélie Nemours', 'Berto Lardera',
           'Charles Lapicque', 'Charmion Von Wiegand', 'César Domela', 'Ellsworth Kelly',
           'Emilio Vedova', 'Fernand Léger', 'František Kupka', 'Franz Kline',
           'François Morellet', 'Georges Mathieu', 'Georges Vantongerloo',
           'Gustave Singier', 'Hans Hartung', 'Jean Arp', 'Jean Bazaine', 'Jean Degottex',
           'Jean Dubuffet', 'Jean Fautrier', 'Jean Gorin', 'Joan Mitchell',
           'Josef Albers', 'Kenneth Noland', 'Leon Polk Smith', 'Lucio Fontana',
           'László Moholy-Nagy', 'Léon Gischia', 'Maria Helena Vieira da Silva',
           'Mark Rothko', 'Morris Louis', 'Naum Gabo', 'Olle Bærtling', 'Otto Freundlich',
           'Pierre Soulages', 'Pierre Tal Coat', 'Piet Mondrian', 'Richard Paul Lohse',
           'Roger Bissière', 'Sam Francis', 'Sonia and Robert Delaunay', 'Sophie Taeuber-Arp',
           'Theo van Doesburg', 'Vassily Kandinsky', 'Victor Vasarely', 'Yves Klein', 'Étienne Béothy')


def download_delauny_original_data(extract_to: Path = Path('~/data/delauny_original_data/'),
                                   path_2_zip=Path('~/data/delauny_original_data/'),
                                   url_all: str = 'https://physiologie.unibe.ch/supplementals/delaunay.zip',
                                   url_train: str = 'https://physiologie.unibe.ch/supplementals/delaunay_train.zip',
                                   url_test: str = 'https://physiologie.unibe.ch/supplementals/delaunay_test.zip',
                                   url_img_urls: str = 'https://physiologie.unibe.ch/supplementals/DELAUNAY_URLs.zip',
                                   ):
    """
    Downloads the abstract art delauny data set for ML and other research.

python -u ~/ultimate-utils/ultimate-utils-proj-src/uutils/torch_uu/dataset/delaunay_uu.py
nohup python -u ~/ultimate-utils/ultimate-utils-proj-src/uutils/torch_uu/dataset/delaunay_uu.py > delauny.out &

    ref: https://github.com/camillegontier/DELAUNAY_dataset/issues/2
    """
    extract_to: Path = expanduser(extract_to)
    download_and_extract(url_all, path_used_for_zip=path_2_zip, path_used_for_dataset=extract_to)
    download_and_extract(url_train, path_used_for_zip=path_2_zip, path_used_for_dataset=extract_to)
    download_and_extract(url_test, path_used_for_zip=path_2_zip, path_used_for_dataset=extract_to)
    download_and_extract(url_img_urls, path_used_for_zip=path_2_zip, path_used_for_dataset=extract_to)

    # urls = [url_all, url_train, url_test, url_img_urls]
    # # - download data (could be made faster with mp or asyncio, whatever)
    # for url in urls:
    #     download_and_unzip(url, extract_to)


def process_delanauny_into_pickle_files():
    """not going to do it for now, just keep the folder per image setup. Effortless. Unless it becomes a problem."""
    pass
    # NOP


def _original_data_transforms_delauny(size: int = 256) -> tuple[Compose, Compose, Compose]:
    """

    Note:
        - original delauny basically only uses resize.
        - they also only have a train split in their data set so they only have a train and test transform. But its
        not a big deal we can put all three transforms the same here.

    ref:
        - https://github.com/camillegontier/DELAUNAY_dataset/blob/main/CNN_training/training.py#L26
    """
    train_data_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std),
    ])
    validation_data_transform = train_data_transform
    test_data_transform = train_data_transform
    return train_data_transform, validation_data_transform, test_data_transform


def _force_to_size_data_transforms_delauny(size_out: int = 84, size: int = 256, padding: int = 8) -> tuple[
    Compose, Compose, Compose]:
    """
    Forces img to be of size_out no matter what

    padding comment: going to keep it since that's what MI does. But usually it's not needed, mostly added when zoom/resize
    screws up with your imag or including both padding and NONE padding to get more data e.g. to make it robust to padding/frames.
    """
    train_data_transform = Compose([
        Resize((size, size)),
        RandomCrop(size_out, padding=padding),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])
    test_data_transform = transforms.Compose([
        Resize((size_out, size_out)),
        transforms.ToTensor(),
        Normalize(mean=mean, std=std),
    ])
    validation_data_transform = test_data_transform
    return train_data_transform, validation_data_transform, test_data_transform


def data_transform_based_on_random_resized_crop_yxw(size: int = 84,
                                                    scale: tuple[int, int] = (0.18, 1.0),
                                                    padding: int = 8,
                                                    ratio: tuple[float, float] = (0.75, 1.3333333333333333),
                                                    ):
    """
    Applies an approximation to the MI data augmentation using RandomResizedCrop.

    Details & decisions:
    Since we want to data difference btw MI & delauny to be due to the data and not data transform we approximate what
    MI does as much as possible. They only crop to 84 with padding to give 84. We do something similar be resize the
    image first (according to the percentages of image resizing of Mini-Imagenet which they do for all images 0.18
    since they do 469 -> 84 which is 0.1781, not imagenet does 469 -> 256 ~ 0.328).
    RandomCrop says that it pads with the given int on both sides: "If a single int is provided this is used to pad all borders."
    so since MI uses 8 but output is 84 it means the size of the actual image is 84 - 8 - 8 on both height and width --
    so we will imitate this.
    Mini imagenet only normalizes:         test_data_transforms = Compose([normalize,]). And yes, it does seem that
    the padding size is added to both sides. Check code check_that_padding_is_added_on_both_sides_so_in_one_dim_it_doubles_the_size
    by searching in pycharm to find it. It passes the size tests I'd expect.
    """
    train_data_transform = Compose([
        RandomResizedCrop((size - padding * 2, size - padding * 2), scale=scale, ratio=ratio),
        Pad(padding=padding),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])
    test_data_transform = transforms.Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])
    validation_data_transform = test_data_transform
    return train_data_transform, validation_data_transform, test_data_transform


def get_my_delauny_data_transforms(data_augmentation: str = 'delauny_uu',
                                   size: int = 84,
                                   ) -> tuple[Compose, Compose, Compose]:
    """

    Notes:
        - RandomCrop has padding = 8 because this likely makes it more robust against images with surrounding contours/padding.
        - val transform == test transform because: I agree to use test transforms on validation. It reduces the variance on the validation and since your not fitting them anyway there is likely little benefit to early stop with complicated train data augmentation for valitation. Better to have a low variance estimate of an unknown distribution so to early stop more precisely.

    ref:
        - https://github.com/learnables/learn2learn/issues/309
        - padding for random crop discussion: https://datascience.stackexchange.com/questions/116201/when-to-use-padding-when-randomly-cropping-images-in-deep-learning
    """
    print(f'{size=} for my delauny.')
    if data_augmentation is None or data_augmentation == 'original_delauny_84':
        train_data_transform, validation_data_transform, test_data_transform = _original_data_transforms_delauny(84)
    elif data_augmentation == 'original_delauny':
        train_data_transform, validation_data_transform, test_data_transform = _original_data_transforms_delauny(
            size=256)
    elif data_augmentation == 'resize256_then_random_crop_to_84_and_padding_8':
        train_data_transform, validation_data_transform, test_data_transform = \
            _force_to_size_data_transforms_delauny(size_out=84, size=256, padding=8)
    elif data_augmentation == 'delauny_random_resized_crop_yxw_padding_8':
        # this one is for training model only on delauny, when combined with other data sets we might need to rethink
        train_data_transform, validation_data_transform, test_data_transform = \
            data_transform_based_on_random_resized_crop_yxw(padding=8)
    elif data_augmentation == 'hdb_mid_mi_delauny':
        # either reuse delauny_random_resized_crop_yxw or do what you wrote to derek hoiem.
        # if the diversity looks high enough we won't implement the idea sent to Derek Hoiem.
        """
        The output is always 84 x 84. So yes it would upsample 46 to 84 (using Resize(84, 616)) and get a [84, 616] image (no padding). Then it would do a normal RandomCrop. This would be an alternative to doing:
torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=InterpolationMode.BILINEAR, antialias: Optional[bool] = None
which scales both dimensions without "control". My goal is to make the data augmentation most similar to what is being done in another data set to make comparisons of the role of the data more fair (and to minimize difference in performance due to data augmentation). 
        """
        pass
        raise NotImplementedError
    elif data_augmentation == 'delauny_random_resized_crop_yxw_zero_padding':
        # this one is for training model only on delauny, when combined with other data sets we might need to rethink
        train_data_transform, validation_data_transform, test_data_transform = \
            data_transform_based_on_random_resized_crop_yxw(padding=0)
    else:
        raise ValueError(f'Err: {data_augmentation=}')
    return train_data_transform, validation_data_transform, test_data_transform


def get_delauny_dataset_splits(path2train: str,
                               path2val: str,
                               path2test: str,
                               data_augmentation: str = 'delauny_uu',
                               size: int = 84,
                               random_split: bool = False,
                               ) -> tuple[Dataset, Dataset, Dataset]:
    """ """
    # - expand paths
    path2train: Path = expanduser(path2train)
    path2val: Path = expanduser(path2val)
    path2test: Path = expanduser(path2test)
    # - data transforms
    train_data_transform, validation_data_transform, test_data_transform = get_my_delauny_data_transforms(
        data_augmentation, size)
    # - get normal pytorch data set with data transforms already in it
    train_dataset = ImageFolder(path2train, transform=train_data_transform)
    if random_split:
        print(f'printing path2val since your using random split, make sure its the empty string: {str(path2val)=}')
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset,
                                                                     [7362, 1840],
                                                                     generator=torch.Generator().manual_seed(42))
        assert str(path2val) != '', f'Err: you have a path2val but we are randomly splitting: {path2val=}'
    else:
        valid_dataset = ImageFolder(path2val, transform=validation_data_transform)
    test_dataset = ImageFolder(path2test, transform=test_data_transform)
    return train_dataset, valid_dataset, test_dataset


def create_your_splits(path_to_all_data: Union[str, Path],
                       path_for_splits: Union[str, Path],
                       ):
    """
    Details on my Delauny few-shot learning data set splits:
    - 34, 8, 11
    - split is deterministic (and arbitrary) based on sorting -- to guarantee determinisim. Hopefully this is diverse enough,
    the diversity for this arbitrary split is: mu +- ci.
    - make sure .labels is set i.e. the 34, 8, 11 & asserts there.
    """
    path_to_all_data: Path = expanduser(path_to_all_data)
    path_for_splits: Path = expanduser(path_for_splits)
    # - get path to union of all images & sort based on alphabetical path to folder [likely first name] (might be useful for usl!)
    dirpath, dirnames, filenames = next(iter(os.walk(path_to_all_data)))
    print(f'{dirpath=}, {path_to_all_data=}')
    print(f'{dirnames=}')
    assert len(dirnames) == 53
    # - split into 34, 8, 11 splits (based on previous sorting list)
    sorted_dirnames: list = list(sorted(dirnames))
    print(f'{sorted_dirnames=}')
    train_val = sorted_dirnames[:42]
    train = train_val[:34]
    val = train_val[34:]
    test = sorted_dirnames[42:]
    assert len(train) == 34
    assert len(val) == 8
    assert len(test) == 11
    # - save the few-shot learning 34, 8, 11 splits as folders with images (based on previous sorting list)
    path2train, path2val, path2test = get_l2l_bm_split_paths(path_for_splits)
    copy_folders_recursively(src_root=path_to_all_data, root4dst=path2train, dirnames4dst=train)
    copy_folders_recursively(src_root=path_to_all_data, root4dst=path2val, dirnames4dst=val)
    copy_folders_recursively(src_root=path_to_all_data, root4dst=path2test, dirnames4dst=test)
    # print the paths to the 3 splits. Check them manually (or print ls to them and print the lst)
    print(f'{path2train=}')
    print(f'{path2val=}')
    print(f'{path2test=}')
    # assert the splits at dst for fsl/lsl are the right sizes e.g. 34, 8, 11
    assert len(next(iter(os.walk(path2train)))[1]) == 34
    assert len(next(iter(os.walk(path2val)))[1]) == 8
    assert len(next(iter(os.walk(path2test)))[1]) == 11
    # - later, compute the task2vec div of the train 34 and test 11 splits.
    print('computing delauny div for this split')
    from uutils.argparse_uu.common import create_default_log_root
    from diversity_src.experiment_mains.main_diversity_with_task2vec import \
        compute_div_and_plot_distance_matrix_for_fsl_benchmark
    from uutils.argparse_uu.meta_learning import parse_args_meta_learning
    from uutils import setup_wandb

    args: Namespace = parse_args_meta_learning()
    args: Namespace = diversity_ala_task2vec_delauny_resnet18_pretrained_imagenet(args)
    setup_wandb(args)
    create_default_log_root(args)
    compute_div_and_plot_distance_matrix_for_fsl_benchmark(args, show_plots=False)


def diversity_ala_task2vec_delauny_resnet18_pretrained_imagenet(args: Namespace) -> Namespace:
    args.batch_size = 5
    args.data_option = 'delauny_uu_l2l_bm_split'
    args.data_path = Path('~/data/delauny_l2l_bm_splitss').expanduser()

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.run_name = f'{args.experiment_name} {args.batch_size=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def get_l2l_bm_split_paths(path_for_splits: Path) -> tuple[Path, Path, Path]:
    path_for_splits: Path = expanduser(path_for_splits)
    path2train: Path = path_for_splits / 'delauny_train_split_dir'
    path2val: Path = path_for_splits / 'delauny_validation_split_dir'
    path2test: Path = path_for_splits / 'delauny_test_split_dir'
    return path2train, path2val, path2test


# - tests

def diversity_ala_task2vec_delauny_resnet18_pretrained_imagenet(args: Namespace) -> Namespace:
    # args.batch_size = 500
    args.batch_size = 2
    args.data_option = 'delauny'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'delauny'

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.experiment_name = f'diversity_ala_task2vec_mi_resnet18'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_option} {args.model_option} {current_time}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def loop_raw_pytorch_delauny_dataset_with_my_data_transforms_and_print_min_max_size():
    path2train: str = '~/data/delauny_original_data/DELAUNAY_train'
    path2val: str = ''
    path2test: str = '/Users/brandomiranda/data/delauny_original_data/DELAUNAY_test'
    random_split = True
    train_dataset, valid_dataset, test_dataset = get_delauny_dataset_splits(path2train, path2val, path2test,
                                                                            random_split=random_split)
    train_loader: DataLoader = DataLoader(train_dataset, num_workers=1)
    valid_loader: DataLoader = DataLoader(valid_dataset, num_workers=1)
    test_loader: DataLoader = DataLoader(test_dataset, num_workers=1)
    next(iter(train_loader))
    next(iter(valid_loader))
    next(iter(test_loader))
    # -
    concat = ConcatDataset([train_dataset, valid_dataset, test_dataset])
    assert len(concat) == len(train_dataset) + len(valid_dataset) + len(test_dataset)
    for i, (x, y) in enumerate(concat):
        print(f'{x=}')
        print(f'{y=}')
        print(f'{x.size()=}')
        print(f'{x.norm()=}')
        print(f'{x.norm()/3=}')
        assert x.size(0) == 3
        break
    # assert i == len(concat)
    # - print min & max sizes
    print('decided not to print it since the current data transform went through all the images without issues')


def loop_my_delauny_based_on_my_disjoint_splits_for_fsl_but_normal_dataloader():
    pass


def create_my_fsl_splits_from_original_delauny_splits():
    path_to_all_data: str = '~/data/delauny_original_data/DELAUNAY'
    path_for_splits: str = '~/data/delauny_l2l_bm_splitss'
    create_your_splits(path_to_all_data, path_for_splits)


if __name__ == "__main__":
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    # download_delauny_original_data()
    # create_my_fsl_splits_from_original_delauny_splits()
    loop_raw_pytorch_delauny_dataset_with_my_data_transforms_and_print_min_max_size()
    # loop_my_delauny_based_on_my_disjoint_splits_for_fsl_but_normal_dataloader()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
