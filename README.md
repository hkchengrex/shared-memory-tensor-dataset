# Shared Memory Tensor Dataset with torchrun

## Overview
- This repository provides an example of reading from a single shared memory tensor from multiple processes (e.g., with DDP).
- Useful for loading a large tensor (e.g., the entire dataset) to the CPU to speed up I/O without incurring Nx memory usage where N is the number of GPUs/processes
- We use the standard `torch.utils.data.Dataloader` which might make it easier for you to use this in your own code
- Works with `torchrun`
- Does not depend on `detectron2`

## Limitation
- We did not test this script in the multi-node settings. It probably would not work.

## Usage

(`N` is the number of GPUs/processes)
1. Run `torchrun --standalone --nproc_per_node=N main-multigpu-naive.py`
2. Look at the memory usage.

3. Run `torchrun --standalone --nproc_per_node=N main-multigpu-shared.py`
4. Look at the memory usage again.

### Dependencies
* Python >= 3.7
* Linux
* PyTorch >= 1.10
* `python -m pip install psutil tabulate`

## Acknowledgement

Inspired by and modified from https://github.com/ppwwyyxx/RAM-multiprocess-dataloader

See also: 
- https://github.com/ppwwyyxx/RAM-multiprocess-dataloader/issues/5
- https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/

