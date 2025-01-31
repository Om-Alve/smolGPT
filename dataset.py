from collections import defaultdict
from itertools import islice
import random
from pathlib import Path
import glob
from typing import Iterator, Tuple
import torch
import numpy as np
from more_itertools import chunked


class PreTokDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size_for_max_seq_len: int, split: str, max_seq_len: int, max_shards: int=None, chunk_ratios=None): #np.arange(0.0, 1.1, 0.1)
        """
        Keyword arguments: 
        chunk_ratios: list[float] -- if set, batching is done with variable chunk sizes. The list of floats are used to calculate these variable chunk sizes by multiplying with the `max_seq_len`. Hence, the maximum ratio value (the max value for an element in this list of floats) can be `1.0`. If `chunk_ratios` is specified, we expect to do variable size chunks sometimes less than the `max_seq_len` - this means we can pack more into a single batch. We calculate how much more based on the chunk size (directly proportional). If the chunk size is half of the `max_seq_len` then we can pack twice the number of batches.
        batch_size_for_max_seq_len: int -- for chunks of size `max_seq_len`, this is the batch size. But if `chunk_ratios` are specified, we may adjust the batch size based on the chunk size (we'll pack more for smaller chunk sizes)
        max_shards - useful for quick debugging. Simply stop yielding after we reach `max_shards` number of shards
        """
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.max_shards = max_shards
        self.chunk_ratios = chunk_ratios
        self.batch_size_for_max_seq_len = batch_size_for_max_seq_len

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        bin_dir = Path("data/TinyStories_all_data")
        shard_filenames = sorted(glob.glob(str(bin_dir / "*.bin")))
        shard_filenames = (
            shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        )

        rng = random.Random(42)
        while True:
            rng.shuffle(shard_filenames)
            for shard in islice(shard_filenames, self.max_shards):
                data = np.memmap(shard, dtype=np.uint16, mode="r")
                # `chunk_ratios` is set - we can calculate a variety of chunk sizes based on this ratio
                if self.chunk_ratios:
                    chunk_sizes = (self.max_seq_len * self.chunk_ratios).astype(int)
                else:
                # no `chunk_ratios` - we only use one chunk size i.e. the `max_seq_len`
                    chunk_sizes = [self.max_seq_len]
                chunk_start = 0
                data_len = len(data)
                chunks = defaultdict(lambda: [])
                while chunk_start < data_len:
                    # sample the `chunk_size_selected` every time based on the available `chunk_sizes`
                    chunk_size_selected = int(np.random.choice(chunk_sizes))
                    chunk_end = chunk_size_selected + chunk_start
                    chunk = torch.from_numpy(data[chunk_start:chunk_end].astype(np.int64))
                    chunk_start = chunk_end
                    # store the chunks in a dictionary that segregates different chunk sizes
                    chunks[chunk_size_selected].append(chunk)
                
                # iterate through the different chunk sizes and the available chunks of that size
                for chunk_size, chunks_of_same_size in chunks.items():
                    # how much can we pack -- depends on the batch size specified for the maximum sequence length. If the chunk size is less than `max_seq_len`, we can pack a lot more
                    num_chunks_of_same_size = self.max_seq_len // chunk_size * self.batch_size_for_max_seq_len
                    # shuffle the chunks to randomize
                    rng.shuffle(chunks_of_same_size)
                    # `chunked` moves the window by `num_chunks_of_same_size` and gives us a list
                    # if we run out of chunks, we use the leftover in one batch -- leads to less packing but shouldn't happen too frequently, only once per chunk_size per shard
                    # TODO: maybe pack similar `chunk_size` batches across shards?
                    for chunks_in_one_batch in chunked(chunks_of_same_size, num_chunks_of_same_size):
                        # stack the inputs and outputs and return the stacked aka. batched output
                        x_to_stack = []
                        y_to_stack = [] 
                        for chunk in chunks_in_one_batch:
                            # all of these chunks will be of the same size `chunk_size`
                            x = chunk[:-1]
                            y = chunk[1:]
                            x_to_stack.append(x)
                            y_to_stack.append(y)
                        # stack them all
                        X = torch.stack(x_to_stack)
                        Y = torch.stack(y_to_stack)
                        # since this is already stacked, the caller can simply use this as a single batch
                        yield X, Y


class Task:
    @staticmethod
    def iter_batches(
        batch_size: int, device: str, num_workers: int = 0, **dataset_kwargs
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        ds = PreTokDataset(batch_size_for_max_seq_len=batch_size, **dataset_kwargs)
        # no batch_size = no batching. Because the `PreTokDataset` already batches data with variable sized chunks
        # based on the chunk size, the batch size is modified
        # `batch_size` input to this method assumes constant size batches with `max_seq_len`
        # instead we reduce `max_seq_len` for some batches and appropriately increase batch size
        # we could just directly use the dataset `ds` to iterate but using the dataloader allows us to set number of workers
        # TODO: does `num_workers` have any use now since we are sequentially processing the shards and chunks? Was there ever a point in having this?
        dl = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            yield x, y
