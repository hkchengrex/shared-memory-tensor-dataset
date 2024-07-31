#!/usr/bin/env python
import itertools
import os
import torch

import torch.distributed as dist

from common import MemoryMonitor
from dataset import TorchTensorDataset


def main():

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    print(f"local_rank={local_rank}, world_size={world_size}")

    monitor = MemoryMonitor()
    ds = TorchTensorDataset(num_gbs=40)
    print(monitor.table())

    loader = torch.utils.data.DataLoader(ds, num_workers=32)

    pids = [os.getpid()]
    all_pids = [None for _ in range(world_size)]
    dist.all_gather_object(all_pids, pids)
    all_pids = list(itertools.chain.from_iterable(all_pids))
    monitor = MemoryMonitor(all_pids)
    
    for _ in range(100):
        for d in loader:
            d = d.cuda()

            if local_rank == 0:
                print(monitor.table())

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
