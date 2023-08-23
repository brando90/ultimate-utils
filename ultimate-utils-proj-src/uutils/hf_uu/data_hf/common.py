from typing import Union

from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset


def get_guanaco_datsets_add_splits_train_test_only(dataset_name: str = "timdettmers/openassistant-guanaco",
                                                   as_tuple: bool = True,
                                                   ) -> Union[dict, tuple]:
    """
This dataset is a subset of the Open Assistant dataset, which you can find here: https://huggingface.co/datasets/OpenAssistant/oasst1/tree/main

This subset of the data only contains the highest-rated paths in the conversation tree, with a total of 9,846 samples.

This dataset was used to train Guanaco with QLoRA.

    ref: https://huggingface.co/datasets/timdettmers/openassistant-guanaco
    """
    trainset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset] = load_dataset(dataset_name,
                                                                                               split="train")
    # valset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset] = load_dataset(dataset_name, split="val")
    valset = None
    testset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset] = load_dataset(dataset_name,
                                                                                              split="test")
    # return
    if as_tuple:
        datasets: tuple = trainset, valset, testset
    else:
        datasets: dict = dict(train=trainset, val=valset, test=testset)
    return datasets
