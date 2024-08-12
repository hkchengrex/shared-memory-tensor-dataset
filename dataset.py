from typing import Any, Optional
import tempfile
import torch
import torch.distributed as dist
from tensordict import MemoryMappedTensor

import os
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

class TorchTensorDataset(torch.utils.data.Dataset):
    def __init__(self, num_gbs=40):
        self.data = torch.randn((1024, 1024, 256, num_gbs)) + 100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TorchSharedTensorDataset(TorchTensorDataset):
    def __init__(self, is_rank0, num_gbs=40):

        if is_rank0:
            super().__init__(num_gbs)
            self.tmp_file = tempfile.NamedTemporaryFile(prefix='shared-tensor-', dir='/dev/shm')
            print(f"Rank {local_rank} created file {self.tmp_file.name}")
            filename = self.tmp_file.name
            # immediately unlink the file; the processes should still have a reference
            os.unlink(filename)
            meta_information = (filename, self.data.shape, self.data.dtype)
        else:
            meta_information = None

        filename, data_shape, data_type = local_scatter_torch(meta_information)

        if is_rank0:
            self.data = MemoryMappedTensor.from_tensor(self.data, filename=filename, existsok=True)
        else:
            self.data = MemoryMappedTensor.from_filename(filename=filename, dtype=data_type, 
                    shape=data_shape)

        dist.barrier()
        


def local_scatter_torch(obj: Optional[Any]):
    if world_size == 1:
        # Just one worker. Do nothing.
        return obj

    array = [obj] * world_size
    target_array = [None]
    if local_rank == 0:
        dist.scatter_object_list(target_array, scatter_object_input_list=array, src=0)
    else:
        dist.scatter_object_list(target_array, scatter_object_input_list=None, src=0)
    return target_array[0]
