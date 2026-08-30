import os
import logging
import warnings
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist

from utils.misc import is_main_process

def _load_state_dict(model, state_dict, verbose=True, logger=None):
    model_keys = model.state_dict().keys()
    missing_in_model = set(state_dict.keys()) - set(model_keys)
    missing_in_weight = set(model_keys) - set(state_dict.keys())

    # Check name-mismatched keys
    if verbose:
        for key in missing_in_weight:
            logger.warn(
                "[MODEL_RESTORE] missing keys in checkpoint: %s" % (key)
            )
        for key in missing_in_model:
            logger.warn("[MODEL_RESTORE] missing keys in model: %s" % (key))

    matched_keys = set(state_dict.keys()).intersection(model_keys)
    assert len(matched_keys) > 0, (
        "Unable to find enough keys matched with model. This means the model "
        "has lots of mismatched keys, please check your model."
    )

    # Check shape-mismacthed keys
    for key in matched_keys:
        assert state_dict[key].size() == model.state_dict()[key].size(), (
            f"Found shape-mismachted key: {key}\t"
            f"weight size: {state_dict[key].size()}\t"
            f"model size: {model.state_dict()[key].size()}"
        )

    model.load_state_dict(state_dict, strict=False)


def restore_model(
    model: nn.Module, model_path: str, logger: logging.Logger, verbose=True
):
    if model_path is None or not os.path.exists(model_path):
        warnings.warn(
            f"[MODEL_RESTORE] Path to weights is invalid: {model_path}"
        )
        return
    if logger is not None:
        logger.info(f"[MODEL_RESTORE] Restoring weights from {model_path}")
    checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage.cpu()
    )
    if isinstance(checkpoint, dict) and "epoch" in checkpoint and logger is not None:
        logger.info(f"[MODEL_RESTORE] checkpoint epoch: {checkpoint['epoch']}")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # support pytorch2 / DDP prefixes
    state_dict = {
        k.replace("_orig_mod.module.", "")
        if k.startswith("_orig_mod.module.")
        else k: v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("module.", "") if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }

    if hasattr(model, "module"):
        model = model.module
    _load_state_dict(model, state_dict, verbose=verbose and logger is not None, logger=logger)


def process_checkpoint(model, optimizer, workspace, epoch, create_symlink = True):
    ckpt_filename = str(epoch + 1)
    if not is_main_process():
        return

    # save checkpoint
    ckpt_path = os.path.join(workspace, ckpt_filename)
    save_checkpoint(model, optimizer, ckpt_path, epoch)
    # create link to latest ckpt
    if create_symlink:
        link_ckpt_path = os.path.join(workspace, "latest.pth")
        shutil.copy(ckpt_path, link_ckpt_path)

def save_checkpoint(model, optimizer, ckpt_path, epoch):
    model_without_ddp = model
    if hasattr(model_without_ddp, "module"):
        model_without_ddp = model_without_ddp.module
    optimizer = (
        optimizer.state_dict()
    )

    state_dict: dict = model_without_ddp.state_dict()
    state_keys = list(state_dict.keys())
    for state_key in state_keys:
        if "ema" in state_key and "." not in state_key:
            state_dict.pop(state_key)

    save_meta = dict(
        epoch=epoch + 1, state_dict=state_dict, optimizer=optimizer
    )
    torch.save(save_meta, ckpt_path)