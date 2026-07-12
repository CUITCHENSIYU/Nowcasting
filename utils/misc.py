import os
import warnings

import torch
import torch.distributed as dist


def init_distributed_mode(args):
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.local_rank = getattr(args, "local_rank", 0) or 0

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.local_rank = args.rank % torch.cuda.device_count()
        args.world_size = torch.cuda.device_count()
    else:
        if torch.cuda.is_available():
            gpu_ids = getattr(args, "gpus", ["0"])
            torch.cuda.set_device(int(gpu_ids[0]))
        warnings.warn("Not using distributed mode")
        return

    args.distributed = True
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    args.dist_backend = "nccl"
    dist.init_process_group(
        backend=args.dist_backend,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()


def is_distributed():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_distributed():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()
