# Copyright (c) Facebook, Inc. and its affiliates.
"""
List serialization code adopted from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py
"""

from typing import List, Any, Optional
import multiprocessing as mp
import torch
import torch.distributed as dist

import os
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

class TorchTensorDataset(torch.utils.data.Dataset):
    def __init__(self, num_gbs=40):
        self.data = torch.zeros((1024, 1024, 256, num_gbs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TorchSharedTensorDataset(TorchTensorDataset):
    def __init__(self, is_rank0, num_gbs=40):
        if is_rank0:
            super().__init__(num_gbs)

        if is_rank0:
            handles = [None] + [
              bytes(mp.reduction.ForkingPickler.dumps(self.data))
              for _ in range(world_size - 1)]
        else:
            handles = None

        handle = local_scatter_torch(handles)

        if local_rank > 0:
            # Materialize the tensor from shared memory.
            self.data = mp.reduction.ForkingPickler.loads(handle)
            print(f"Worker {local_rank} obtains a dataset of length="
                  f"{len(self)} from its local leader.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def local_scatter_torch(array: Optional[List[Any]]):
    """
    Scatter an array from local leader to all local workers.
    The i-th local worker gets array[i].

    Args:
        array: Array with same size of #local workers.
    """
    if world_size == 1:
        # Just one worker. Do nothing.
        return array[0]

    target_array = [None]
    if local_rank == 0:
        assert len(array) == world_size
        dist.scatter_object_list(target_array, scatter_object_input_list=array, src=0)
    else:
        dist.scatter_object_list(target_array, scatter_object_input_list=None, src=0)
    return target_array[0]

