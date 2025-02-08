import torch
import numpy as np
import random
from pathlib import Path
import glob
from typing import Iterator, Tuple
import torch.distributed as dist

class PreTokDataset(torch.utils.data.IterableDataset):
    def __init__(self, split: str, max_seq_len: int):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = 42 + worker_id + 1337 * rank
        bin_dir = Path("data/TinyStories_all_data")
        shard_filenames = sorted(glob.glob(str(bin_dir / "*.bin")))
        shard_filenames = (
            shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        )
        rng = random.Random(seed)
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                data = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(data) // self.max_seq_len - 1
                idxs = list(range(num_batches))
                rng.shuffle(idxs)

                for idx in idxs:
                    start = idx * self.max_seq_len
                    end = (idx + 1) * self.max_seq_len
                    chunk = torch.from_numpy(data[start:end].astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class Task:
    @staticmethod
    def iter_batches(
        batch_size: int, device: str, num_workers: int = 0, **dataset_kwargs
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        ds = PreTokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            yield x, y

