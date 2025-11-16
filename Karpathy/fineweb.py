
"""
download fineweb-edu 10B Tokens
preproces pre tokenize***
save data shards to disk


tokenize the document:
    * gpt2 encoder
    * start with [eot] token (end of text, but start with it)
    * extend tokens with text
    * np array the tokens
    * assert tokens fit into 16bit int
    * convert to np.uint16
    * return np.uitn16 tokens

    * tokenize all doc
    * write output to shards .npy
    *


10B tokens * 2bytes (uint16) = ~20GB
fineweb-edu/sample/10BT = 28.5 GB
48.5GB
goes on ~/.cache/huggingface/datasets

check free space on SSD
`df -h` or `df -h ~`

"""
import os
from datasets import load_dataset
import tiktoken
import numpy as np
import multiprocessing as mp
from tqdm import tqdm


local_dir = 'edu_fineweb10B'
remote_name = 'sample-10BT'
shard_size = int(1e8) # 100M tokens per shards. so 100 shards for 10B tokens

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download hf dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# tokenize
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']
def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2 **16).all(), 'token dict too big for uint16'
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


# mp to write shards
nprocs = max(1, os.cpu_count()//2) # return half number of cpu cores

"""
SINGLE CORE map()
    map(fn, iterable):
        returns an iterator that applies the function to every item in iterable
        yields results

    def map_from_scratch(fn, iterator):
        for i in iterator:
            yield fn(i)

MULTICORE mp.Pool.map()
    with mp.Pool(processes=nproc) as pool:
        result = pool.map(fn, iterable, chunksize=100) # sens 100 items per worker tasks
    print(result)

    Problem:
        converts iterable of tiems into list
        then sibmits all items as tasks to process pool
        but blocks until all paralle ltasks are doen and the full ist of results is ready

IMAP()
    submit takss one by one to the process pool and retrives results as they are completed
    items yielded one at a time instead of all at once like map()
    yielded in order of completion instead of after all task are completed

example:
    iterable = [1,2,3,4,5,6,7,8]
    chunksize = 2
    nproc = 4
    chunks into 8/2 = [1,2], [3,4], [5,6], [7,8]
    each worker processes one chunk

"""


with mp.Pool(nprocs) as pool: # create nprocs separate python processes to run in parallel
    #pool object provides map, imap, apply_sync to distribute work to workers
    shard_index = 0
    # preallocate buffer for writing in shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    # imap issues tasks to process pool
    # pool(fn, iterable)
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # each worker handles 16 doc per task
        # tokenizes them within own process, and returns a list of tokens, creates shards, next chunk

        if token_count + len(tokens)<= shard_size: # enough space for tokens in current shard?
            # append tokens to shard buffer
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)

            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit='tokens', desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))

        else:
            # write current shard & start new shard buffer with leftover tokens
            split = 'val' if shard_index == 0 else 'train'
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}") # pad 6 digits leading 0
            # how many tokens fit in current shard
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index+=1
            progress_bar = None
            # populate new shard with leftovers
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # write any remaining tokens as a last shard
    if token_count != 0:
        split = 'val' if shard_index == 0 else 'train'
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
