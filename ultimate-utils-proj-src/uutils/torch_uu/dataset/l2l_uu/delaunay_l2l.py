"""
Outline of how to create a l2l taskloader:
- for now see https://github.com/learnables/learn2learn/issues/375

note:
    - to create a task loader from a new data set: https://github.com/learnables/learn2learn/issues/375
    - remember we have a l2l to torchmeta loader converter. So for now all code is written in l2l's api.
"""


def get_delauny_l2l_datasets_and_task_transforms(train_ways=5,
                                                 train_samples=10,
                                                 test_ways=5,
                                                 test_samples=10,
                                                 root='~/data',
                                                 data_augmentation=None,
                                                 device=None,
                                                 **kwargs,
                                                 ):
    """Tasksets for mini-ImageNet benchmarks."""
    if data_augmentation is None:
        train_data_transforms = None
        test_data_transforms = None
    elif data_augmentation == 'normalize':
        train_data_transforms = Compose([
            lambda x: x / 255.0,
        ])
        test_data_transforms = train_data_transforms
    elif data_augmentation == 'lee2019':
        normalize = Normalize(
            mean=[120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
            std=[70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0],
        )
        train_data_transforms = Compose([
            ToPILImage(),
            RandomCrop(84, padding=8),  # todo: do we really need th padding = 8 for delauny, check notes, check l2l git issues if not post one
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])
        test_data_transforms = Compose([
            normalize,
        ])
    else:
        raise ValueError('Invalid data_augmentation argument.')

    from uutils.torch_uu.dataset.delaunay_uu import get_my_delauny_dataset_splits
    train_dataset, valid_dataset, test_dataset = get_my_delauny_dataset_splits()
    # train_dataset = l2l.vision.datasets.MiniImagenet(
    #     root=root,
    #     mode='train',
    #     download=True,
    # )
    # valid_dataset = l2l.vision.datasets.MiniImagenet(
    #     root=root,
    #     mode='validation',
    #     download=True,
    # )
    # test_dataset = l2l.vision.datasets.MiniImagenet(
    #     root=root,
    #     mode='test',
    #     download=True,
    # )
    # - todo, do we need this? my delauny data set pytorch thing already has it, check with data augmentation
    if device is None:
        train_dataset.transform = train_data_transforms
        valid_dataset.transform = test_data_transforms
        test_dataset.transform = test_data_transforms
    else:
        # todo, do we need this?
        train_dataset = l2l.data.OnDeviceDataset(
            dataset=train_dataset,
            transform=train_data_transforms,
            device=device,
        )
        valid_dataset = l2l.data.OnDeviceDataset(
            dataset=valid_dataset,
            transform=test_data_transforms,
            device=device,
        )
        test_dataset = l2l.data.OnDeviceDataset(
            dataset=test_dataset,
            transform=test_data_transforms,
            device=device,
        )
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    # compare task transforms with omniglots
    train_transforms = [
        NWays(train_dataset, train_ways),
        KShots(train_dataset, train_samples),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    valid_transforms = [
        NWays(valid_dataset, test_ways),
        KShots(valid_dataset, test_samples),
        LoadData(valid_dataset),
        ConsecutiveLabels(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    test_transforms = [
        NWays(test_dataset, test_ways),
        KShots(test_dataset, test_samples),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]

    _datasets = (train_dataset, valid_dataset, test_dataset)
    _transforms = (train_transforms, valid_transforms, test_transforms)
    return _datasets, _transforms


def get_delauny_tasksets():
    root = os.path.expanduser(root)

    # Load task-specific data and transforms
    datasets, transforms = get_delauny_l2l_datasets_and_task_transforms(train_ways=train_ways,
                                                                        train_samples=train_samples,
                                                                        test_ways=test_ways,
                                                                        test_samples=test_samples,
                                                                        root=root,
                                                                        device=device,
                                                                        **kwargs)
    train_dataset, validation_dataset, test_dataset = datasets
    train_transforms, validation_transforms, test_transforms = transforms

    # Instantiate the tasksets
    train_tasks = l2l.data.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )
    validation_tasks = l2l.data.TaskDataset(
        dataset=validation_dataset,
        task_transforms=validation_transforms,
        num_tasks=num_tasks,
    )
    test_tasks = l2l.data.TaskDataset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=num_tasks,  # todo: hu?! remove this
    )
    return BenchmarkTasksets(train_tasks, validation_tasks, test_tasks)


# -- Run experiment

def loop_through_hdb4_delaunay():
    print(f'test: {loop_through_hdb4_delaunay=}')
    # - for determinism
    # random.seed(0)
    # torch.manual_seed(0)
    # np.random.seed(0)
    #
    # # - options for number of tasks/meta-batch size
    # batch_size = 2
    #
    # # - get benchmark
    # benchmark: BenchmarkTasksets = hdb3_mi_omniglot_delauny_tasksets()
    # splits = ['train', 'validation', 'test']
    # tasksets = [getattr(benchmark, split) for split in splits]
    #
    # # - loop through tasks
    # device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    # # from models import get_model
    # # model = get_model('resnet18', pretrained=False, num_classes=5).to(device)
    # # model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V2")
    # # model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    # from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
    # model, _ = get_default_learner_and_hps_dict()  # 5cnn
    # model.to(device)
    # criterion = nn.CrossEntropyLoss()
    # for i, taskset in enumerate(tasksets):
    #     print(f'-- {splits[i]=}')
    #     for task_num in range(batch_size):
    #         print(f'{task_num=}')
    #
    #         X, y = taskset.sample()
    #         print(f'{X.size()=}')
    #         print(f'{y.size()=}')
    #         print(f'{y=}')
    #
    #         y_pred = model(X)
    #         loss = criterion(y_pred, y)
    #         print(f'{loss=}')
    #         print()
    print(f'done test: {loop_through_hdb4_delaunay=}')


if __name__ == "__main__":
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    loop_through_hdb4_delaunay()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
