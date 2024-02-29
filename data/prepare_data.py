# prepare openwebtext data for training
# from project root run: python -m data.prepare_data

import os
from tqdm import tqdm
import numpy as np
from llama.tokenizer import Tokenizer
from datasets import load_dataset

num_proc = 32
enc = Tokenizer("llama/models/tokenizer.model")
save_data_path = 'data/datasets/openwebtext'

if __name__ == "__main__":
    data = load_dataset("openwebtext", num_proc=num_proc)
    train_val_dataset = data["train"].train_test_split(test_size=0.0005, seed=42, shuffle=True)
    train_val_dataset['val'] = train_val_dataset.pop('test') # rename test to val
    os.makedirs(save_data_path, exist_ok=True)

    def process_dataset(example):
        text = example['text']
        tokens = enc.encode(text, bos=True, eos=True)
        return {"ids": tokens, "len": len(tokens)}

    # tokenize the data
    tokenized_data = train_val_dataset.map(process_dataset, remove_columns=["text"], num_proc=num_proc)

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
