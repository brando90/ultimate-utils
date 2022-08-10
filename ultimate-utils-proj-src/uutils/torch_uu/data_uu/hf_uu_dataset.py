from pathlib import Path
from typing import Union

import pandas as pd
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

from datasets import load_dataset, DatasetDict, Dataset

from uutils import expanduser
from uutils.torch_uu import approx_equal


def get_data_set_with_splits(all_data: Dataset,
                             train_size: float = 0.8,
                             validation_size: float = 0.1,
                             test_size: float = 0.1,
                             seed: int = 0,
                             ) -> DatasetDict:
    original_size: int = len(all_data)
    assert approx_equal(train_size + validation_size + test_size, 1.0, tolerance=1e-3)
    train_val_test: DatasetDict = all_data.train_test_split(train_size=train_size + validation_size, seed=seed)

    frac_of_train: float = train_size / (train_size + validation_size)  # e.g. 0.9x = 0.8 === (t + v*x) = real_train
    frac_of_val: float = validation_size / (train_size + validation_size)
    assert approx_equal(frac_of_train + frac_of_val, 1.0, tolerance=1e-3)
    train_val: DatasetDict = train_val_test['train'].train_test_split(train_size=frac_of_train, seed=seed)

    train: Dataset = train_val['train']
    validation: Dataset = train_val['test']
    test: Dataset = train_val_test['test']
    dataset: Dataset = DatasetDict(train=train, validation=validation, test=test)
    err_msg = f"Expected: {(len(dataset['train']) + len(dataset['validation']) + len(dataset['test']))=} " \
              f"But got: {original_size=}"
    assert len(dataset['train']) + len(dataset['validation']) + len(dataset['test']) == original_size, err_msg
    return dataset


def get_dataset_from_json_file(path2filename: Path) -> Dataset:
    """

    note: hf Dataset seems to handle recursive data with dicts of dicts fine! At least if its just a json structure.
    See the ps field/key/column is a dict:
        dataset=Dataset({
            features: ['tt', 'tt_id', 'proof_step_id', 'theorem', 'ps', 'ppt', 'ptp', 'tactic', 'stmt_id', 'len_tac_proof', 'hts', 'ept', 'r_ppt', 'r_ept', 'coq_proj', 'coq_package_pin', 'filename', 'goal_names_short', 'goal_names_long', 'unique_filename', 'regex_hts', 'error'],
            num_rows: 854
        })
        dataset[0]={'tt': '\n\nExample test_next_weekday:\n  (next_weekday (next_weekday saturday)) = tuesday.\n', 'tt_id': 3, 'proof_step_id': 0, 'theorem': '', 'ps': {'local_ctx_goals': 'none\n============================\nnext_weekday (next_weekday saturday) =\n                             tuesday', 'global_ctx': 'none\n', 'local_ctx_seperator': '\n============================\n', 'goals_seperator': '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n', 'global_ctx_seperator': '\n----------------------------\n'}, 'ppt': '?Goal', 'ptp': ' \n\nExample test_next_weekday:\n  (next_weekday (next_weekday saturday)) = tuesday.\n', 'tactic': '\n\nExample test_next_weekday:\n  (next_weekday (next_weekday saturday)) = tuesday.\n', 'stmt_id': 3, 'len_tac_proof': 3, 'hts': ['eq_refl'], 'ept': 'eq_refl', 'r_ppt': '(__hole <<<< 0 ?Goal >>>>)', 'r_ept': '(__hole <<<< 0 (__hole <<<< 1 eq_refl >>>>) >>>>)', 'coq_proj': 'lf', 'coq_package_pin': '/home/bot/pycoq/pycoq/test/lf', 'filename': '/home/bot/.opam/ocaml-variants.4.07.1+flambda_coq-serapi.8.11.0+0.11.1/.opam-switch/build/lf.dev/Basics.v._pycoq_context', 'goal_names_short': ['?Goal'], 'goal_names_long': ['?Goal'], 'unique_filename': 'data_point_0.json', 'regex_hts': [], 'error': False}
        dataset[0]
    """
    expanduser(path2filename)
    try:
        dataset: Dataset = Dataset.from_json(path2filename)
    except AttributeError:
        dataset: Dataset = Dataset.from_json(str(path2filename))
    except Exception as e:
        raise e
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
