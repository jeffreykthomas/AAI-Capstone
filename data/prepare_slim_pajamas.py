import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count

import data.packed_dataset as packed_dataset
from llama.tokenizer import Tokenizer

# Filename for SlimPajama
slimpajama_sets = {
	"train": "SlimPajama-627B/train/chunk*/*",
	"validation": "SlimPajama-627B/validation/chunk*/*",
	"test": "SlimPajama-627B/test/chunk*/*",
}


def prepare_full(
		source_path: Path,
		tokenizer_path: str,
		destination_path: Path,
		chunk_size: int,
		split: str = "validation",
		filenames_subset: List[str] = None,
		process_id: int = 0
) -> None:
	import zstandard as zstd

	destination_path.mkdir(parents=True, exist_ok=True)

	tokenizer = Tokenizer(tokenizer_path)

	# Use the provided filenames_subset or default to all filenames
	filenames = filenames_subset

	if not filenames:
		raise RuntimeError(
			f"No files matching {slimpajama_sets[split]} found at {source_path}. \n"
			"Make sure you download the data..."
		)

	builder = packed_dataset.PackedDatasetBuilder(
		outdir=destination_path,
		prefix=f"{split}_slimpajama_{process_id}",  # Use process_id to differentiate builders
		chunk_size=chunk_size,
		sep_token=tokenizer.bos_id,
		dtype="auto",
		vocab_size=tokenizer.sp_model.vocab_size()
	)

	for filepath in filenames:
		print(f"Processing {filepath}")
		with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
			for row in tqdm(f):
				text = json.loads(row)["text"]
				if json.loads(row)["meta"]["redpajama_set_name"] == "RedPajamaGithub":
					continue  # we don't want to include the github data
				text_ids = tokenizer.encode(text, bos=True, eos=False)
				builder.add_array(np.array(text_ids, dtype=builder.dtype))


# we throw away the final corpus to avoid meaningless corpus filled with bos_ids,
# see https://github.com/jzhang38/TinyLlama/issues/83 for more details
# builder.write_reminder()


def prepare(
		source_path: Path = Path("/data/datasets"),
		tokenizer_path: str = "llama/models/tokenizer.model",
		destination_path: Path = Path("/data/datasets/slim-pajama/val"),
		chunk_size: int = 2049 * 1024,
		split: str = "validation",
		percentage: float = 1.0,
) -> None:
	import time

	filenames = glob.glob(os.path.join(source_path, slimpajama_sets[split]), recursive=True)
	filenames = filenames[:int(len(filenames) * percentage)]

	num_processes = cpu_count()
	chunked_filenames = np.array_split(filenames, num_processes)

	processes = []
	start_time = time.time()

	for i, subset in enumerate(chunked_filenames):
		p = Process(
			target=prepare_full,
			args=(source_path, tokenizer_path, destination_path, chunk_size, split, list(subset), i)
		)
		processes.append(p)
		p.start()

	for p in processes:
		p.join()
	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
	from jsonargparse import CLI

	CLI(prepare)
