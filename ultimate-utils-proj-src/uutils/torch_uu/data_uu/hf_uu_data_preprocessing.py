from typing import Union, Callable

import datasets
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast


def tokenize_function(tokenizer, examples: datasets.arrow_dataset.Batch):
    """

    notes:
    - used to preprocess input
    - seems that padding can be done later, see trans preprocess func
    """
    encoded_batch: BatchEncoding = tokenizer(examples["text"], padding="max_length", truncation=True)
    batch_size: int = len(examples['text'])
    assert batch_size == len(examples['label'])
    # encode_batch = tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")
    # return tokenizer(examples["text"], padding="max_length", truncation=True)
    # print(encoded_batch)
    return encoded_batch  # e.g. {'input_ids': [[101, 173, 1197, 119, 22, ...}


def preprocess_function_translation_tutorial(examples: datasets.arrow_dataset.Batch,
                                             tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                                             prefix: str = "translate English to French: ",
                                             source_lang: str = 'en',
                                             target_lang: str = 'fr',
                                             ) -> BatchEncoding:
    """

    note:
    - padding and other stuff done at DataCollatorForSeq2Seq

    Inspections:
    examples
        Out[6]: {'id': ['12513', '55647'], 'translation': [{'en': 'Descending the laurel walk, I faced the wreck of the chestnut-tree; it stood up black and riven: the trunk, split down the centre, gasped ghastly.', 'fr': "Après avoir descendu l'allée de lauriers, je regardai le marronnier frappé par la foudre."}, {'en': '"He said, \'Come, here\'s a ladder that\'s of no use!\' and he took it."', 'fr': 'Il a dit : Tiens, voilà une échelle qui ne sert pas ! et il l’a prise. »'}]}

    model_inputs
        Out[6]: {'input_ids': [[13959, 1566, 12, 2379, 10, 451, 3, 25390, 1891, 140, 82, 123, 15, 5, 1],
        [13959, 1566, 12, 2379, 10, 37, 3, 1765, 24943, 248, 3553, 21, 82, 2353, 6, 113, 141, 9717, 44, 8, 680, 397, 13, 3, 28150, 9, 7, 6, 11, 8, 7117, 47, 7020, 5, 1]],
        'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
    labels
        Out[5]: {'input_ids': [[19024, 6, 4679, 50, 197, 77, 2693, 9620, 3, 9, 13847, 721, 6, 3, 1558, 142, 3, 60, 5884, 295, 3, 9, 50, 12, 699, 6, 50, 2629, 21379, 4530, 93, 8446, 5, 1], [1636, 4116, 11891, 3, 55, 5495, 11891, 3, 55, 22669, 17, 90, 1072, 9, 2498, 29, 3, 18, 285, 3, 16187, 288, 1394, 7632, 12449, 3, 15, 17, 1394, 11387, 3, 26, 31, 154, 900, 1938, 6, 769, 159, 8870, 245, 13715, 27849, 7, 343, 2317, 3, 18, 3, 15, 17, 3, 40, 31, 106, 5931, 12524, 155, 20, 303, 10490, 171, 10407, 49, 197, 13498, 3, 55, 1]],
        'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
    """
    # approximately batch of sequences, but sequences are still strings
    inputs: list[str] = [prefix + example[source_lang] for example in examples["translation"]]
    targets: list[str] = [example[target_lang] for example in examples["translation"]]
    # encodes strings to the token ids & returns other useful stuff like the attention make (see comments in fun def).
    model_inputs: BatchEncoding = tokenizer(inputs, max_length=128, truncation=True)

    # tokenize targets with special mode for target e.g. targets might need to be right shifted while encoder might not etc.
    with tokenizer.as_target_tokenizer():  # Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to sequence-to-sequence models that need a slightly different processing for the labels.
        labels: BatchEncoding = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def helper_get_preprocess_function_translation_tutorial(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                                                        prefix: str = "translate English to French: ",
                                                        source_lang: str = 'en',
                                                        target_lang: str = 'fr',
                                                        verbose: bool = False,
                                                        ) -> Callable:
    """
    Gets the preprocess function since the dataset.map function takes a function that only takes in examples as input.
    """
    if verbose:
        print(f'{isinstance(tokenizer, PreTrainedTokenizer)=}')
        print(f'{isinstance(tokenizer, PreTrainedTokenizerFast)=}')
    f = lambda examples: preprocess_function_translation_tutorial(examples, tokenizer, prefix, source_lang, target_lang)
    return f


#

def preprocess_function_translation_tutorial(examples: datasets.arrow_dataset.Batch,
                                             tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                                             prefix: str = "Generate entire proof term: ",
                                             ) -> BatchEncoding:
    """

    note:
    - padding and other stuff done at DataCollatorForSeq2Seq
    """
    # approximately batch of sequences, but sequences are still strings
    inputs: list[str] = [prefix + example['ptp'] for example in examples]
    targets: list[str] = [example['ept'] for example in examples]

    # encodes strings to the token ids & returns other useful stuff like the attention make (see comments in fun def).
    model_inputs: BatchEncoding = tokenizer(inputs, max_length=128, truncation=True)

    # tokenize targets with special mode for target e.g. targets might need to be right shifted while encoder might not etc.
    with tokenizer.as_target_tokenizer():  # Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to sequence-to-sequence models that need a slightly different processing for the labels.
        labels: BatchEncoding = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# - tests

if __name__ == '__main__':
    print('Done!\a')
