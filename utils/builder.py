from typing import Optional, Union, List

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.optim as optim

from utils.registry import get_module
from utils.misc import is_distributed
from utils.checkpoint import restore_model


def overrite_default_args(args: dict, default_args: dict):
    for key, value in default_args.items():
        if isinstance(value, dict):
            if key not in args or not isinstance(args[key], dict):
                args[key] = {}
            overrite_default_args(args[key], value)
        else:
            args[key] = value


def build_module(
    config: Optional[dict],
    parent: str,
    split: Optional[str] = None,
    default_args: Optional[Union[dict, List[dict]]] = None,
):
    if config is None or not isinstance(config, dict):
        return None

    if "type" not in config:
        config_str = "\n".join([f" - {k}: {v}" for k, v in config.items()])
        raise KeyError(
            "config must contain the key 'type' to choose module, "
            f"but got: \n{config_str}",
        )

    args = config.copy()
    if split is not None and split in args and isinstance(args[split], dict):
        split_overrides = args.pop(split)
        args.update(split_overrides)
    elif split is not None:
        for key, value in args.items():
            if not isinstance(value, dict):
                continue
            value_dict = value.copy()
            for split_key, split_value in value_dict.items():
                if split_key == split:
                    args[key] = split_value

    if default_args is not None:
        if isinstance(default_args, dict):
            overrite_default_args(args, default_args)
        elif isinstance(default_args, list):
            for _default_args in default_args:
                if _default_args is not None:
                    overrite_default_args(args, _default_args)

    try:
        module = None
        module_type = args.pop("type")
        if module_type is not None:
            module = get_module(parent, module_type)(**args)
        return module
    except Exception as e:
        raise type(e)(
            f"build module failed for [{module_type}] in "
            f"{parent} parent because: {e}"
        )


def build_dataloader(
    dataset: Union[Dataset, dict],
    batch_size: int,
    num_workers: int,
    split="train",
    default_args: Optional[dict] = None,
):
    if isinstance(dataset, dict):
        dataset_cfg = dataset.copy()
        dataset = build_module(
            dataset_cfg,
            parent="datasets",
            split=split,
            default_args=default_args,
        )

    collate_fn = getattr(dataset, "collate_fn", None)
    shuffle = split == "train" and not is_distributed()
    sampler = None
    if is_distributed():
        sampler = DistributedSampler(dataset, shuffle=(split == "train"))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=shuffle,
    )
    return dataloader


def build_model(
    model_cfg: dict,
    device: torch.device,
    logger=None,
    default_args: Optional[Union[list, dict]] = None,
    pretrained_weight_path: Optional[str] = None,
    find_unused_parameters: bool = False,
):
    assert isinstance(model_cfg, dict)
    model: nn.Module = build_module(
        model_cfg, parent="models", default_args=default_args
    )
    model = model.to(device)
    # Load on every rank so DDP workers start from the same weights.
    if pretrained_weight_path is not None:
        restore_model(model, pretrained_weight_path, logger=logger)

    if is_distributed():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.index is not None else None,
            output_device=device.index,
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=False,
        )
    return model


def build_optimizer(
    optimizer_cfg: dict,
    model: nn.Module,
    default_args: Optional[dict] = None,
):
    optimizer_cfg = optimizer_cfg.copy()
    optim_type = optimizer_cfg.pop("type")
    lr = optimizer_cfg.pop("lr")
    weight_decay = optimizer_cfg.pop("weight_decay", 0.0)
    optim_params = optimizer_cfg.pop("params", {})
    if default_args is not None:
        optim_params.update(**default_args)

    if hasattr(model, "module"):
        model = model.module

    params = [{"params": model.parameters(), "lr": lr, "weight_decay": weight_decay}]
    optimizer: optim.Optimizer = getattr(optim, optim_type)(params, **optim_params)
    return optimizer


def build_criterion(criterion_cfg: dict, default_args: Optional[dict] = None):
    assert isinstance(criterion_cfg, dict)
    return build_module(criterion_cfg, parent="criterions", default_args=default_args)


def build_scheduler(optimizer, scheduler_cfg: dict, max_epochs: int, steps_per_epoch: int):
    scheduler_cfg = scheduler_cfg.copy()
    scheduler_type = scheduler_cfg.pop("type", "OneCycleLR")
    if scheduler_type == "OneCycleLR":
        lrs = [group["lr"] for group in optimizer.param_groups]
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            lrs,
            epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            base_momentum=0.85,
            max_momentum=0.95,
            **scheduler_cfg,
        )
    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
