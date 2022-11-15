"""

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
"""
from datetime import datetime

from argparse import Namespace

import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, Compose, ToPILImage, RandomCrop, ColorJitter, RandomHorizontalFlip, \
    ToTensor

from typing import Union

from pathlib import Path

from uutils import download_and_extract, expanduser, move_folders_recursively

mean = [0.5853, 0.5335, 0.4950]
std = [0.2348, 0.2260, 0.2242]
normalize = transforms.Normalize(mean=mean,
                                 std=std)


# classes = ('Ad Reinhardt', 'Alberto Magnelli', 'Alfred Manessier', 'Anthony Caro',
#             'Antoine Pevsner', 'Auguste Herbin', 'Aurélie Nemours', 'Berto Lardera',
#             'Charles Lapicque', 'Charmion Von Wiegand', 'César Domela', 'Ellsworth Kelly',
#             'Emilio Vedova', 'Fernand Léger', 'František Kupka', 'Franz Kline',
#             'François Morellet', 'Georges Mathieu', 'Georges Vantongerloo',
#             'Gustave Singier', 'Hans Hartung', 'Jean Arp', 'Jean Bazaine', 'Jean Degottex',
#             'Jean Dubuffet', 'Jean Fautrier', 'Jean Gorin', 'Joan Mitchell',
#             'Josef Albers', 'Kenneth Noland', 'Leon Polk Smith', 'Lucio Fontana',
#             'László Moholy-Nagy', 'Léon Gischia', 'Maria Helena Vieira da Silva',
#             'Mark Rothko', 'Morris Louis', 'Naum Gabo', 'Olle Bærtling', 'Otto Freundlich',
#             'Pierre Soulages', 'Pierre Tal Coat', 'Piet Mondrian', 'Richard Paul Lohse',
#             'Roger Bissière', 'Sam Francis', 'Sonia and Robert Delaunay', 'Sophie Taeuber-Arp',
#             'Theo van Doesburg', 'Vassily Kandinsky', 'Victor Vasarely', 'Yves Klein', 'Étienne Béothy')

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


def get_min_max_size_of_images_delany() -> tuple[int, int]:
    """
    Loop through data sets (all images) and collect the min and max sizes. Also print the channels, assert it to be 3.
    """
    pass  # todo


def get_data_augmentation():
    pass  # todo


def _original_data_transforms_delauny(size: int = 256) -> transforms.Compose:
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std),
    ])
    return transform


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
    if data_augmentation is None:
        raise NotImplementedError
        # return original delauny transforms
    elif data_augmentation == 'delauny_uu':
        train_data_transform = Compose([
            ToPILImage(),
            RandomCrop(size, padding=8),
            # decided 8 due to https://datascience.stackexchange.com/questions/116201/when-to-use-padding-when-randomly-cropping-images-in-deep-learning
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])
        test_data_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std),
        ])
        validation_data_transform = test_data_transform
    else:
        raise ValueError(f'Err: {data_augmentation=}')
    return train_data_transform, validation_data_transform, test_data_transform


def get_my_delauny_dataset_splits(path2train: str,
                                  path2val: str,
                                  path2test: str,
                                  size: int = 84,  # todo
                                  ) -> tuple[Dataset, Dataset, Dataset]:
    path2train: Path = expanduser(path2train)
    path2val: Path = expanduser(path2val)
    path2test: Path = expanduser(path2test)
    # Loads the train and test data ###############################################
    train_dataset = ImageFolder(path2train, transform=transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std),
    ]))
    valid_dataset = ImageFolder(path2val, transform=transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std),
    ]))
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset_base,
    #                                                            [7362, 1840],
    #                                                            generator=torch.Generator().manual_seed(42))

    test_dataset = ImageFolder(path2test, transform=transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std),
    ]))
    # todo: check all the data transforms are alright by seeing the mi, in particular check data augmentation for mi and this: https://github.com/camillegontier/DELAUNAY_dataset/issues/3
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
    # - get path to union of all images & sort based on alphabetical path to folder [likely first name] (might be useful for usl!)
    path_to_all_data: Path = expanduser(path_to_all_data)
    dirpath, dirnames, filenames = next(iter(os.walk(path_to_all_data)))
    print(f'{dirpath=}, {path_to_all_data=}')
    # assert dirpath == path_to_all_data
    assert len(dirnames) == 53
    # - split into 34, 8, 11 splits (based on previous sorting list)
    sorted_dirnames: list = list(sorted(dirnames))
    train_val = sorted_dirnames[:42]
    train = train_val[:34]
    val = train_val[34:]
    test = sorted_dirnames[42:]
    assert len(train) == 34
    assert len(val) == 8
    assert len(test) == 11
    # - save the few-shot learning 34, 8, 11 splits as folders with images (based on previous sorting list)
    path_for_splits: Path = expanduser(path_for_splits)
    path2train: Path = path_for_splits / 'delauny_train_split_dir'
    move_folders_recursively(root=path_for_splits / 'delauny_train_split_dir', dirnames=train)
    path2val: Path = path_for_splits / 'delauny_validation_split_dir'
    move_folders_recursively(root=path_for_splits / 'delauny_validation_split_dir', dirnames=val)
    path2test: Path = path_for_splits / 'delauny_test_split_dir'
    move_folders_recursively(root=path_for_splits / 'delauny_test_split_dir', dirnames=test)
    # - print the paths to the 3 splits. Check them manually (or print ls to them and print the lst)
    print(f'{path2train=}')
    print(f'{path2val=}')
    print(f'{path2test=}')
    # - later, compute the task2vec div of the train 34 and test 11 splits.
    # args: Namespace = load_args()
    # args: Namespace = Namespace()
    # args: Namespace = diversity_ala_task2vec_delauny_resnet18_pretrained_imagenet()
    # compute_div_and_plot_distance_matrix_for_fsl_benchmark(args, show_plots=False)
    # ## compute_div_and_plot_distance_matrix_for_fsl_benchmark(args)


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


def loop_raw_pytorch_delauny_dataset():
    pass


if __name__ == "__main__":
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    download_delauny_original_data()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")

    # # %%
    # """
    # Copy pasted, perhaps todo later perhaps, adapt to my library.
    # ref: https://github.com/camillegontier/DELAUNAY_dataset/blob/main/CNN_training/training.py
    # """
    # # -*- coding: utf-8 -*-
    # """
    # Created on Wed Nov 10 19:44:17 2021
    # @author: gontier
    # """
    #
    # # Relevant packages ##########################################################
    #
    # from __future__ import print_function
    # import torch
    # import torch.nn as nn
    # import torch.optim as optim
    # import torchvision.transforms as transforms
    # from torchvision.datasets import ImageFolder
    # import os
    # from torchvision import models
    #
    # torch.cuda.empty_cache()
    # import numpy as np
    # import random
    #
    # # Parameters ##################################################################
    #
    # batch_size = 20
    # nb_epoch = 300
    # size = 256
    # weight_decay = 0.0025
    #
    # torch.manual_seed(1234)
    # np.random.seed(31)
    # random.seed(32)
    # torch.cuda.manual_seed_all(33)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    #
    # # Sets device #################################################################
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    #
    # # Train and test data directory ###############################################
    #
    # data_dir_train = os.getcwd() + "/DELAUNAY_train"
    # data_dir_test = os.getcwd() + "/DELAUNAY_test"
    #
    # # Loads the train and test data ###############################################
    #
    # dataset_base = ImageFolder(data_dir_train, transform=transforms.Compose([
    #     transforms.Resize((size, size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5853, 0.5335, 0.4950],
    #                          std=[0.2348, 0.2260, 0.2242]),
    # ]))
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset_base,
    #                                                            [7362, 1840],
    #                                                            generator=torch.Generator().manual_seed(42))
    #
    # test_dataset = ImageFolder(data_dir_test, transform=transforms.Compose([
    #     transforms.Resize((size, size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5853, 0.5335, 0.4950],
    #                          std=[0.2348, 0.2260, 0.2242]),
    # ]))
    #
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    #     pin_memory=True,
    #     generator=torch.Generator().manual_seed(43)
    # )
    #
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    #     pin_memory=True,
    #     generator=torch.Generator().manual_seed(44)
    # )
    #
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    #     pin_memory=True,
    #     generator=torch.Generator().manual_seed(45)
    # )
    #
    # # Classes ####################################################################
    #
    # classes = ('Ad Reinhardt', 'Alberto Magnelli', 'Alfred Manessier', 'Anthony Caro',
    #            'Antoine Pevsner', 'Auguste Herbin', 'Aurélie Nemours', 'Berto Lardera',
    #            'Charles Lapicque', 'Charmion Von Wiegand', 'César Domela', 'Ellsworth Kelly',
    #            'Emilio Vedova', 'Fernand Léger', 'František Kupka', 'Franz Kline',
    #            'François Morellet', 'Georges Mathieu', 'Georges Vantongerloo',
    #            'Gustave Singier', 'Hans Hartung', 'Jean Arp', 'Jean Bazaine', 'Jean Degottex',
    #            'Jean Dubuffet', 'Jean Fautrier', 'Jean Gorin', 'Joan Mitchell',
    #            'Josef Albers', 'Kenneth Noland', 'Leon Polk Smith', 'Lucio Fontana',
    #            'László Moholy-Nagy', 'Léon Gischia', 'Maria Helena Vieira da Silva',
    #            'Mark Rothko', 'Morris Louis', 'Naum Gabo', 'Olle Bærtling', 'Otto Freundlich',
    #            'Pierre Soulages', 'Pierre Tal Coat', 'Piet Mondrian', 'Richard Paul Lohse',
    #            'Roger Bissière', 'Sam Francis', 'Sonia and Robert Delaunay', 'Sophie Taeuber-Arp',
    #            'Theo van Doesburg', 'Vassily Kandinsky', 'Victor Vasarely', 'Yves Klein', 'Étienne Béothy')

    # CNN ########################################################################

    # net = models.resnet152(pretrained=False)
    # net.to(device)
    #
    # # Loss function and optimizer ################################################
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.003, weight_decay=weight_decay)
    #
    # # Training ###################################################################
    #
    # train_error_values = []
    # val_error_values = []
    #
    # for epoch in range(nb_epoch):
    #
    #     # Train ##################################################################
    #     running_loss = 0.0
    #
    #     for i, data in enumerate(train_loader, 0):
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #
    #         optimizer.zero_grad()
    #
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.item()
    #
    #     running_loss = running_loss / len(train_loader)
    #
    #     # display the epoch training loss
    #     print("epoch : {}/{}, loss_recons = {:.6f}".format(epoch + 1, nb_epoch, running_loss))
    #
    #     # Compute training error #################################################
    #
    #     correct = 0
    #     total = 0
    #
    #     with torch.no_grad():
    #         for data in train_loader:
    #             images, labels = data[0].to(device), data[1].to(device)
    #
    #             outputs = net(images)
    #
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #
    #     print('Accuracy of the network on the training images: %d %%' % (
    #             100 * correct / total))
    #     train_error_values.append(100 - 100 * correct / total)
    #
    #     ###################################################################
    #
    #     # Val ###################################################################
    #     correct = 0
    #     total = 0
    #
    #     with torch.no_grad():
    #         for data in val_loader:
    #             images, labels = data[0].to(device), data[1].to(device)
    #
    #             outputs = net(images)
    #
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #
    #     print('Accuracy of the network on the validation images: %d %%' % (
    #             100 * correct / total))
    #     val_error_values.append(100 - 100 * correct / total)
    #
    # # Test ###################################################################
    #
    # correct = 0
    # total = 0
    #
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #
    #         outputs = net(images)
    #
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # print('Accuracy of the network on the test images: %d %%' % (
    #         100 * correct / total))
    # test_error_values = (100 - 100 * correct / total)
    #
    # print('Finished Training')
    #
    # # Final results ##############################################################
    #
    # correct_pred = {classname: 0 for classname in classes}
    # total_pred = {classname: 0 for classname in classes}
    #
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = net(images)
    #         _, predictions = torch.max(outputs, 1)
    #
    #         for label, prediction in zip(labels, predictions):
    #             if label == prediction:
    #                 correct_pred[classes[label]] += 1
    #             total_pred[classes[label]] += 1
    #
    # accuracy_values = []
    #
    # for classname, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
    #                                                          accuracy))
    #     accuracy_values.append(accuracy)
    #
    # # Confusion matrix ##############################################################
    #
    # y_pred = []
    # y_true = []
    #
    # for inputs, labels in test_loader:
    #     inputs, labels = inputs.to(device), labels.to(device)
    #     output = net(inputs)
    #
    #     output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    #     y_pred.extend(output)
    #
    #     labels = labels.data.cpu().numpy()
    #     y_true.extend(labels)
