# prepare openwebtext data for training
# from project root run: python -m data.prepare_data

import os
import argparse
import random
import glob
from tqdm import tqdm
import numpy as np
from llama.tokenizer import Tokenizer
from datasets import load_dataset
import torch
import torch.distributed as dist
import torch.nn.functional as F


# Create a custom PyTorch Dataset for loading the pretokenized data
class PretokDataset(torch.utils.data.IterableDataset):
    '''Loads pretokenized examples from disk and yields them as PyTorch tensors.'''

    def __init__(self, split, max_seq_len, vocab_size, dataset):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.dataset = dataset

        if self.dataset == 'openwebtext':
            self.save_data_path = 'data/datasets/openwebtext'
        elif self.dataset == 'TinyStories':
            self.save_data_path = 'data/datasets/TinyStories'
        elif self.dataset == 'dialogues':
            self.save_data_path = 'data/datasets/dialogues'
        elif self.dataset == 'mental_health_dialogues':
            self.save_data_path = 'data/datasets/mental_health_dialogues'

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
        bin_dir = self.save_data_path
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
                    pad_length = self.max_seq_len - len(x)
                    pad_token_id = 2
                    if pad_length > 0:
                        x = F.pad(x, (0, pad_length), value=pad_token_id)
                        y = F.pad(y, (0, pad_length), value=pad_token_id)
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


def process_dataset(example, enc=Tokenizer('llama/models/tokenizer.model')):
    if example is None or example['text'] is None:
        print('Skipping example with None text')
        return {'ids': [], 'len': 0}
    text = example['text']
    text = text.strip()
    tokens = enc.encode(text, bos=True, eos=False)
    return {'ids': tokens, 'len': len(tokens)}


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_proc', type=int, default=32)
    argparser.add_argument('--dataset', type=str, default='openwebtext', help='openwebtext or empathetic_dialogues')
    args = argparser.parse_args()

    enc = Tokenizer('llama/models/tokenizer.model')

    # Save the tokenized data to this directory
    if args.dataset == 'openwebtext':
        save_data_path = '/data/datasets/openwebtext'
    elif args.dataset == 'dialogues':
        save_data_path = '/data/datasets/dialogues'
    elif args.dataset == 'mental_health_dialogues':
        save_data_path = '/data/datasets/mental_health_dialogues'
    elif args.dataset == 'TinyStories':
        save_data_path = 'data/datasets/TinyStories'
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    if args.dataset == 'openwebtext':
        data = load_dataset('openwebtext', num_proc=args.num_proc)
        train_val_dataset = data['train'].train_test_split(test_size=0.0005, seed=42, shuffle=True)
        train_val_dataset['val'] = train_val_dataset.pop('test')  # rename test to val
    elif args.dataset == 'TinyStories':
        data = load_dataset('roneneldan/TinyStories', num_proc=args.num_proc)
        train_val_dataset = data['train'].train_test_split(test_size=0.0005, seed=42, shuffle=True)
        train_val_dataset['val'] = train_val_dataset.pop('test')  # rename test to val
    elif args.dataset == 'dialogues':
        data = load_dataset('csv', data_files={
            'train': '/data/datasets/dialogues/train_dataset.csv',
            'val': '/data/datasets/dialogues/val_dataset.csv',
        }, num_proc=args.num_proc)
        train_val_dataset = data
    elif args.dataset == 'mental_health_dialogues':
        data = load_dataset('csv', data_files={
            'train': '/data/datasets/mental_health_dialogues/train_dataset.csv',
            'val': '/data/datasets/mental_health_dialogues/val_dataset.csv',
        }, num_proc=args.num_proc)
        train_val_dataset = data
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')


    def filter_empty_entries(example):
        # Keep the example only if 'len' is greater than 0
        return example['len'] > 0

    # tokenize the data
    tokenized_data = train_val_dataset.map(process_dataset, remove_columns=['text'], num_proc=args.num_proc)

    # Filter out any empty entries
    filtered_data = tokenized_data.filter(filter_empty_entries)
    # Ensure your save path exists
    os.makedirs(save_data_path, exist_ok=True)

    # save the tokenized data
    # Iterate over each split in the tokenized dataset
    for split, dataset in filtered_data.items():
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
        total_batches = 1024 if len(dataset) > 1024 else len(dataset)

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
