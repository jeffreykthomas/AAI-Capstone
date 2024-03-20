# prepare openwebtext data for training
# from project root run: python -m data.prepare_data

import os
import sys
import random
import glob
from tqdm import tqdm
import numpy as np
from llama.tokenizer import Tokenizer
from datasets import load_dataset
import torch
import torch.distributed as dist

num_proc = 32
enc = Tokenizer('llama/models/tokenizer.model')

dataset = 'openwebtext'

if dataset == 'openwebtext':
    save_data_path = 'data/datasets/openwebtext'
else:
    save_data_path = 'data/datasets/slim-pajama-tokenized'
    
'''
cache_dir = os.environ.get('HF_HOME', None)
if cache_dir is None:
    print('Cache directory is not specified. Please set the HF_HOME environment variable or specify a cache_dir.')
    sys.exit(1)

# Check if the directory exists
if not os.path.exists(cache_dir):
    print(f'Cache directory does not exist: {cache_dir}')
    sys.exit(1)

# Check if the directory is writable
if not os.access(cache_dir, os.W_OK):
    print(f'Cache directory is not writable: {cache_dir}')
    sys.exit(1)

# Check if the directory is readable
if not os.access(cache_dir, os.R_OK):
    print(f'Cache directory is not readable: {cache_dir}')
    sys.exit(1)

print(f'Using cache directory: {cache_dir}')
'''

class PretokDataset(torch.utils.data.IterableDataset):
    '''Loads pretokenized examples from disk and yields them as PyTorch tensors.'''

    def __init__(self, split, max_seq_len, vocab_size):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f'\nRank {rank}, Worker {worker_id}: Created a pre-tokenized Dataset with rng seed {seed}')

        # the .bin files are right along the .json files
        bin_dir = save_data_path
        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, '*.bin')))

        # train/val split. train.bin for train, val.bin for val
        shard_filenames = shard_filenames[:1] if self.split == 'train' else shard_filenames[1:]
        assert len(shard_filenames) > 0, f'No bin files found in {bin_dir}'
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode='r')
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, 'this shard is too small, investigate.'
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


# Function to filter out unwanted set names
def filter_fn(example):
    return example['meta']['redpajama_set_name'] not in excluded_sets


if __name__ == '__main__':
    if dataset == 'openwebtext':
        data = load_dataset('openwebtext', num_proc=num_proc)
        train_val_dataset = data['train'].train_test_split(test_size=0.0005, seed=42, shuffle=True)
        train_val_dataset['val'] = train_val_dataset.pop('test')  # rename test to val
    else:
        # Define a list of set names to exclude
        excluded_sets = ['RedPajamaGithub']

        train_data_files_pattern = '/data/datasets/SlimPajama-627B/train/chunk*/**/*.jsonl.zst'
        val_data_files_pattern = '/data/datasets/SlimPajama-627B/validation/chunk*/**/*.jsonl.zst'
        # Load the predefined splits from the dataset
        data_train = (load_dataset(
            'json',
            data_files=train_data_files_pattern,
            split='train',
            num_proc=num_proc,
            keep_in_memory=False)
                      .filter(filter_fn))
        data_val = (load_dataset(
            'json',
            data_files=val_data_files_pattern,
            split='validation',
            num_proc=num_proc,
            keep_in_memory=False)
                    .filter(filter_fn))

        # Concatenate the training and validation datasets
        train_val_dataset = {
            'train': data_train,
            'val': data_val
        }

    # Ensure your save path exists
    os.makedirs(save_data_path, exist_ok=True)


    def process_dataset(example):
        text = example['text']
        text.strip()
        tokens = enc.encode(text, bos=True, eos=False)
        return {'ids': tokens, 'len': len(tokens)}


    # tokenize the data
    tokenized_data = train_val_dataset.map(process_dataset, remove_columns=['text'], num_proc=num_proc)

    # save the tokenized data
    # Iterate over each split in the tokenized dataset
    for split, dataset in tokenized_data.items():
        # Calculate the total length of all tokenized sequences in the dataset
        total_length = np.sum(dataset['len'], dtype=np.uint64)
        filename = os.path.join(save_data_path, f'{split}.bin')
        # Set the data type for storage, chosen based on max token value
        dtype = np.uint16  # Suitable because max token value is less than 2**16
        # Create a memory-mapped array for efficient writing
        tokens_array = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_length,))

        # Initialize the start index for writing to the memory-mapped array
        write_position = 0
        # The number of batches to divide the dataset into for processing
        total_batches = 1024

        # Iterate over each batch, showing progress with tqdm
        for batch_index in tqdm(range(total_batches), desc=f'Writing to {filename}'):
            # Shard the dataset into batches for processing
            batch = dataset.shard(num_shards=total_batches, index=batch_index, contiguous=True).with_format('numpy')
            # Concatenate the 'ids' field from the batch into a single array
            batch_tokens = np.concatenate(batch['ids'])
            # Write the concatenated batch to the memory-mapped file
            tokens_array[write_position:write_position + len(batch_tokens)] = batch_tokens
            # Update the write position for the next batch
            write_position += len(batch_tokens)

        # Ensure all data is written to the file
        tokens_array.flush()
        # Delete the memory-mapped array to free memory
        del tokens_array
