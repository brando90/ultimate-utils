"""

remember we have a l2l to torchmeta loader converter. So for now all code is written in l2l's api.
"""

def download_delauny_original_data(url_all: str = 'https://physiologie.unibe.ch/supplementals/delaunay.zip',
                                   url_train: str = 'https://physiologie.unibe.ch/supplementals/delaunay_train.zip',
                                   url_test: str = 'https://physiologie.unibe.ch/supplementals/delaunay_test.zip',
                                   url_img_urls: str = 'https://physiologie.unibe.ch/supplementals/DELAUNAY_URLs.zip',
                                   ):
    """
    Downloads the abstract art delauny data set for ML and other research.

    ref: https://github.com/camillegontier/DELAUNAY_dataset/issues/2
    """
    pass


def split_delauny_randomly():
    """
    split according to 64 16 20 like mi so 34 8 11
    """
    pass


# -- Run experiment

def download_delauny_original_data():
    """

    download all, train, test and img urls
    """
    pass


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