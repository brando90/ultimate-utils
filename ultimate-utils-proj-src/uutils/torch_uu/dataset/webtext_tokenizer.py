import os
import numpy as np
import tiktoken
from tqdm import tqdm

from datasets import load_dataset, DatasetDict



def get_tokenized_webtext():
    split_name = 'train'
    tokenized_file_loc = '/lfs/mercury1/0/saumg/gpt2/openwebtext_'+split_name+'_tokenised.bin'
    tokenized_dtype = np.uint16

    if os.path.exists(tokenized_file_loc):
        # entire tokenization exists already
        tokenized_mmap = np.memmap(tokenized_file_loc, dtype=tokenized_dtype, mode='r')
        return tokenized_mmap

    else:
        # open an mmap
        # tokenized_mmap = np.memmap(tokenized_file_loc, dtype=tokenized_dtype, mode='w+', shape = (1,))


        num_eles_added = 0
        # perform tokenization on 10% of the data at a time
        for split_perc in range(10):
            # if split_perc == 0:
            #     split_name = 'train[:'+str((split_perc + 1)*10)+'%]'
            # else:
            #     split_name = 'train['+str(split_perc*10)+'%:'+str((split_perc + 1)*10)+'%]'
            split_name = 'train['+str(split_perc*10)+'%:'+str((split_perc + 1)*10)+'%]'
            # split_name = 'train[:1%]'
            split_file_loc = '/lfs/mercury1/0/saumg/gpt2/openwebtext_'+split_name+'_tokenised.bin'

            if not os.path.exists(split_file_loc):
                print("tokenisation starting at split_perc=", split_perc)
                total_dataset = DatasetDict()
                total_dataset['train'] = load_dataset('openwebtext', split = split_name, cache_dir = '/lfs/mercury1/0/saumg/gpt2/')

                enc = tiktoken.get_encoding("gpt2")
                def process(example):
                    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
                    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
                    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
                    out = {'ids': ids, 'len': len(ids)}
                    return out

                # print(split_dataset)
                # tokenize the dataset
                tokenized = total_dataset.map(
                    process,
                    remove_columns=['text'],
                    desc="tokenizing the splits",
                    # num_proc=8, # TODO
                )

                arr_len = np.sum(tokenized['train']['len'])
                # dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
                arr = np.memmap(split_file_loc, dtype=tokenized_dtype, mode='w+', shape=(arr_len,))

                print("writing split file...")
                idx = 0
                for example in tqdm(tokenized['train']):
                    arr[idx : idx + example['len']] = example['ids']
                    idx += example['len']
                arr.flush()
                print("written")
            else:
                arr = np.memmap(split_file_loc, dtype = tokenized_dtype, mode = 'r')

            # add arr object to tokenized_mmap
            if os.path.exists(tokenized_file_loc):
                tokenized_mmap = np.memmap(tokenized_file_loc, dtype=tokenized_dtype, mode='r+', shape = (num_eles_added + arr.size,))
            else:
                tokenized_mmap = np.memmap(tokenized_file_loc, dtype=tokenized_dtype, mode='w+', shape = (num_eles_added + arr.size,))
            print("adding split to total mmap")
            tokenized_mmap[num_eles_added:] = arr
            num_eles_added = tokenized_mmap.size
            print("flushing total mmap")
            tokenized_mmap.flush()
            print("flushed")

        return tokenized_mmap


if __name__ == "__main__":
    get_tokenized_webtext()