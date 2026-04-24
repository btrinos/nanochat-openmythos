"""Small shared helpers for the nanochat OpenMythos experiment."""

import os

import torch
import torch.distributed as dist


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env]
    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
        return torch.bfloat16
    return torch.float32


COMPUTE_DTYPE = _detect_compute_dtype()


def print0(message="", **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(message, **kwargs)


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return True, rank, local_rank, world_size
    return False, 0, 0, 1
