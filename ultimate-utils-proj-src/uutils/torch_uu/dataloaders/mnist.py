from torchvision import datasets

NORMALIZE_MNIST = transforms.Normalize((0.1307,), (0.3081,))  # MNIST


def get_train_valid_loader(data_dir: Path,
                           batch_size,
                           random_seed,
                           augment=False,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MNIST dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    train_kwargs = {'batch_size': args.batch_size}

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            NORMALIZE_MNIST
        ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            NORMALIZE_MNIST
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            NORMALIZE_MNIST
        ])

    # load the dataset
    train_dataset = datasets.MNIST(root=data_dir, train=True,
                download=True, transform=train_transform)

    valid_dataset = datasets.MNIST(root=data_dir, train=True,
                download=True, transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size, sampler=train_sampler,
                    num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                    batch_size=batch_size, sampler=valid_sampler,
                    num_workers=num_workers, pin_memory=pin_memory)


    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=9,
                                                    shuffle=shuffle,
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory)
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        plot_images(X, labels)

    return (train_loader, valid_loader)

def f():
    test_kwargs = {'batch_size': args.test_batch_size}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST(args.path_to_data_set, train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST(args.path_to_data_set, train=False,
                       transform=transform)
    # Loading Data and splitting it into train and validation data
    train = datasets.MNIST('', train=True, transform=transforms, download=True)
    train, valid = random_split(train, [50000, 10000])
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)