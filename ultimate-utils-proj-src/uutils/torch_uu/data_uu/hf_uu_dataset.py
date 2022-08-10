from typing import Union

from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

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
