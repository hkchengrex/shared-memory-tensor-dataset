#!/usr/bin/env python
import itertools
import multiprocessing as mp
import os
import logging
import torch
import torch.distributed as dist

from common import MemoryMonitor
from dataset import TorchSharedTensorDataset

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])


logger = logging.getLogger(__name__)

def local_broadcast_process_authkey():
    if world_size == 1:
        return
    authkey = bytes(mp.current_process().authkey)
    all_keys = [None for _ in  range(world_size)]
    dist.all_gather_object(all_keys, authkey)
    local_leader_key = all_keys[0]
    if authkey != local_leader_key:
        print('Overwriting local authkey...')
        mp.current_process().authkey = local_leader_key


def main():
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    local_broadcast_process_authkey()

    print(f"local_rank={local_rank}, world_size={world_size}")

    monitor = MemoryMonitor()
    ds = TorchSharedTensorDataset(is_rank0=(local_rank == 0), num_gbs=40)
    print(monitor.table())

    loader = torch.utils.data.DataLoader(ds,
                                         num_workers=0,
                                         shuffle=False,
                                         batch_size=1)

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

            dist.barrier()
            logger.warning(f'{local_rank}, {d.min()}, {d.max()}')  # just make sure the data is correct
            dist.barrier()

    dist.destroy_process_group()
    print('done')

if __name__ == "__main__":
    main()
