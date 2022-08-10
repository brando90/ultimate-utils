from pathlib import Path
from typing import Union

import pandas as pd
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

from datasets import load_dataset, DatasetDict, Dataset

from uutils import expanduser


def get_data_set_books_tutorial(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None
                                ):
    # https://huggingface.co/docs/transformers/tasks/translation
    import datasets
    from datasets import load_dataset, DatasetDict
    books: DatasetDict = load_dataset("opus_books", "en-fr")
    print(f'{books=}')
    books: DatasetDict = books["train"].train_test_split(test_size=0.2)
    print(f'{books=}')
    print(books["train"][0])
    """
    {'id': '90560',
     'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
      'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}
    """
    if tokenizer is not None:
        # - t5 tokenizer
        from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer

        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("t5-small")
        print(f'{isinstance(tokenizer, PreTrainedTokenizer)=}')
        print(f'{isinstance(tokenizer, PreTrainedTokenizerFast)=}')
    else:
        raise NotImplementedError

    # Use 🤗 Datasets map method to apply a preprocessing function over the entire dataset:
    # todo - would be nice to remove this since gpt-2/3 size you can't preprocess the entire data set...or can you?
    # tokenized_books = books.map(preprocess_function, batched=True, batch_size=2)
    from uutils.torch_uu.data_uu.hf_uu_data_preprocessing import helper_get_preprocess_function_translation_tutorial
    preprocessor = helper_get_preprocess_function_translation_tutorial(tokenizer)
    tokenized_books = books.map(preprocessor, batched=True, batch_size=2)
    return tokenized_books


def get_dataset_from_json_file(path2filename: Path) -> Dataset:
    """ """
    expanduser(path2filename)
    dataset: Dataset = Dataset.from_json(path2filename)
    return dataset

# - toy data set

def get_toy_data() -> Dataset:
    """
    Very useful to know thata HF datasets retain the table format that pandas has:
df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                   columns=['a', 'b', 'c'])
df2
   a  b  c
0  1  2  3
1  4  5  6
2  7  8  9
    """
    # two data points, with features ptp, ept as columns
    toy_data = [
        [
            " \n\nTheorem plus_comm: forall n m: nat, n + m = m + n.\n Proof.\n   intros n m.  induction n as [|n1 IHn1].\n   - rewrite <- plus_n_O.  simpl.  reflexivity.\n   - rewrite <- plus_n_Sm.  simpl.  rewrite -> IHn1. ",
            "(fun n m : nat =>\n nat_ind (fun n0 : nat => n0 + m = m + n0)\n   (eq_ind m (fun n0 : nat => 0 + m = n0) eq_refl (m + 0) (plus_n_O m))\n   (fun (n1 : nat) (IHn1 : n1 + m = m + n1) =>\n    eq_ind (S (m + n1)) (fun n0 : nat => S n1 + m = n0)\n      (eq_ind_r (fun n0 : nat => S n0 = S (m + n1)) eq_refl IHn1) \n      (m + S n1) (plus_n_Sm m n1)) n)"],
        [
            " \n\n\nTheorem plus_assoc: forall n m p: nat, n + (m + p) = (n + m) + p.\n   intros n m p.  induction n as [| n1 IHn1].\n   - simpl.  reflexivity.\n   - simpl.  rewrite -> IHn1.  reflexivity.\n",
            "(fun n m p : nat =>\n nat_ind (fun n0 : nat => n0 + (m + p) = n0 + m + p) eq_refl\n   (fun (n1 : nat) (IHn1 : n1 + (m + p) = n1 + m + p) =>\n    eq_ind_r (fun n0 : nat => S n0 = S (n1 + m + p)) eq_refl IHn1) n)"]
    ]
    columns = ['ptp', 'ept']
    # print(toy_data)
    toy_data_pd: pd.DataFrame = pd.DataFrame(toy_data, columns=columns)
    print(toy_data_pd)
    dataset: Dataset = Dataset.from_pandas(toy_data_pd)
    return dataset


def get_toy_dataset_custom() -> DatasetDict:
    # books: DatasetDict = load_dataset("opus_books", "en-fr")
    train: Dataset = get_toy_data()
    test: Dataset = get_toy_data()
    datasets: DatasetDict = DatasetDict(train=train, test=test)
    # validation: Dataset = get_toy_data()

    datasets_clean: DatasetDict = datasets["train"].train_test_split(train_size=0.5, seed=42)
    # Rename the default "test" split to "validation"
    datasets_clean["validation"] = datasets_clean.pop("test")
    datasets_clean["test"] = datasets["test"]

    # datasets: DatasetDict = DatasetDict(train=train, validation=validation, test=test)
    return datasets_clean

def expt():
    Dataset.from_json('path/to/dataset.json')

# - tests

def print_toy_dict_dataset():
    datasets: DatasetDict = get_toy_dataset_custom()
    print(datasets)

    datasets = load_dataset("opus_books", "en-fr")
    print(datasets)
    print(datasets['train'])
    print('print all of the translation pairs & the datapoint id')
    print(f"{len(datasets['train'])=}")
    print('print a data point')
    print(datasets['train'][0])

if __name__ == '__main__':
    print_toy_dict_dataset()
    print('Done\a')