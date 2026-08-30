import os
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.predrnn.rnn import reshape_patch
from utils.builder import build_dataloader, build_model, build_optimizer, build_scheduler
from utils.checkpoint import process_checkpoint
from utils.logger import get_logger
from utils.misc import is_distributed, is_main_process
from utils.registry import register_module


@register_module(parent="runners")
class RNNTrainer:
    """Trainer for PredRNN radar nowcasting with scheduled sampling."""

    def __init__(
        self,
        dataset: dict,
        model: dict,
        workspace: str,
        criterion: Optional[dict] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        max_epochs: int = 50,
        val_interval: int = 5,
        enable_amp: bool = False,
        log_interval: int = 20,
        pretrained_weight_path: Optional[str] = None,
        resume: bool = False,
        find_unused_parameters: bool = False,
        optimizer: Optional[dict] = None,
        scheduler: Optional[dict] = None,
        # scheduled sampling
        scheduled_sampling: bool = True,
        sampling_stop_iter: int = 50000,
        sampling_changing_rate: float = 0.00002,
        eta: float = 1.0,
        log_file: str = "training.log",
        local_rank: int = 0,
        world_size: int = 1,
        **kwargs,
    ):
        self.workspace = workspace
        self.local_rank = local_rank
        self.world_size = world_size

        os.makedirs(self.workspace, exist_ok=True)
        log_file = os.path.join(self.workspace, log_file)
        self.logger = get_logger("predrnn", log_file=log_file) if is_main_process() else None
        self.tensorboard = SummaryWriter(self.workspace) if is_main_process() else None

        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        # PredRNN ST-LSTM + LayerNorm is safer in fp32 by default.
        self.enable_amp = enable_amp and torch.cuda.is_available()
        self.log_interval = log_interval
        self.global_step = 0
        self.epoch = 0

        self.scheduled_sampling = scheduled_sampling
        self.sampling_stop_iter = sampling_stop_iter
        self.sampling_changing_rate = sampling_changing_rate
        self.eta = float(eta)

        dataset_cfg = dataset
        model_cfg = model

        self.seq_len = int(dataset_cfg["seq_len"])
        self.forecast_steps = int(dataset_cfg["forecast_steps"])
        self.total_length = self.seq_len + self.forecast_steps
        self.patch_size = int(model_cfg.get("patch_size", 4))
        self.img_channel = int(model_cfg.get("img_channel", 1))
        self.img_size = int(model_cfg.get("img_size", dataset_cfg.get("input_size", 64)))

        model_cfg = model_cfg.copy()
        model_cfg.setdefault("total_length", self.total_length)
        model_cfg.setdefault("img_size", self.img_size)

        self.train_dataloader: DataLoader = build_dataloader(
            dataset_cfg,
            batch_size=batch_size,
            num_workers=num_workers,
            split="train",
        )
        self.val_dataloader: Optional[DataLoader] = None
        if "val" in dataset_cfg:
            self.val_dataloader = build_dataloader(
                dataset_cfg,
                batch_size=batch_size,
                num_workers=num_workers,
                split="val",
            )

        if resume and not pretrained_weight_path:
            raise ValueError("resume=True requires pretrained_weight_path to a checkpoint")

        self.model = build_model(
            model_cfg,
            device=self.device,
            logger=self.logger,
            pretrained_weight_path=pretrained_weight_path,
            find_unused_parameters=find_unused_parameters,
        )
        self.optimizer = build_optimizer(optimizer, self.model)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)
        self.scheduler = None
        if scheduler is not None:
            self.scheduler = build_scheduler(
                self.optimizer,
                scheduler,
                max_epochs=max_epochs,
                steps_per_epoch=len(self.train_dataloader),
            )

        if resume:
            checkpoint = torch.load(pretrained_weight_path, map_location="cpu")
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "epoch" in checkpoint:
                self.epoch = int(checkpoint["epoch"])
            if self.logger:
                self.logger.info(
                    f"Resumed training from {pretrained_weight_path}, "
                    f"next epoch={self.epoch}"
                )

    def _set_train_epoch(self, epoch: int):
        if is_distributed():
            self.train_dataloader.sampler.set_epoch(epoch)

    def _unwrap_model(self):
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        return model

    def _schedule_sampling(self, batch_size: int, train: bool) -> torch.Tensor:
        """Build mask_true in patch space: [B, T-1, H, W, C]."""
        patch_h = self.img_size // self.patch_size
        patch_w = self.img_size // self.patch_size
        patch_ch = self.img_channel * (self.patch_size ** 2)

        context_masks = self.seq_len - 1
        forecast_masks = self.total_length - self.seq_len
        # (seq_len-1) + forecast_steps = total_length-1

        ones = np.ones(
            (batch_size, context_masks, patch_h, patch_w, patch_ch), dtype=np.float32
        )

        if (not train) or (not self.scheduled_sampling):
            zeros = np.zeros(
                (batch_size, forecast_masks, patch_h, patch_w, patch_ch),
                dtype=np.float32,
            )
            mask = np.concatenate([ones, zeros], axis=1)
            return torch.from_numpy(mask)

        if self.global_step < self.sampling_stop_iter:
            self.eta = max(self.eta - self.sampling_changing_rate, 0.0)
        else:
            self.eta = 0.0

        random_flip = np.random.random_sample(
            (batch_size, forecast_masks, patch_h, patch_w, patch_ch)
        )
        true_token = (random_flip < self.eta).astype(np.float32)
        mask = np.concatenate([ones, true_token], axis=1)
        return torch.from_numpy(mask)

    def _forward_batch(self, frames: torch.Tensor, train: bool):
        batch_size = frames.shape[0]
        frames_patch = reshape_patch(frames, self.patch_size)
        mask_true = self._schedule_sampling(batch_size, train=train).to(
            frames.device, non_blocking=True
        )

        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            pred_patch, loss = self.model(frames_patch, mask_true)
        return pred_patch, loss

    def train_epoch(self, epoch: int):
        self.model.train()
        self._set_train_epoch(epoch)
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_dataloader):
            frames = batch["frames"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            pred_patch, loss = self._forward_batch(frames, train=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {loss.item()}")

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            loss_val = float(loss.detach().item())
            epoch_loss += loss_val
            num_batches += 1
            self.global_step += 1

            if is_main_process() and (batch_idx + 1) % self.log_interval == 0:
                elapsed = time.time() - start_time
                if self.logger:
                    self.logger.info(
                        f"Epoch [{epoch + 1}/{self.max_epochs}] "
                        f"Iter [{batch_idx + 1}/{len(self.train_dataloader)}] "
                        f"loss={loss_val:.4f} eta={self.eta:.4f} time={elapsed:.1f}s"
                    )
                if self.tensorboard:
                    self.tensorboard.add_scalar("train/loss", loss_val, self.global_step)
                    self.tensorboard.add_scalar("train/eta", self.eta, self.global_step)

            del pred_patch, loss

        return epoch_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, epoch: int):
        if self.val_dataloader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            frames = batch["frames"].to(self.device, non_blocking=True)
            _, loss = self._forward_batch(frames, train=False)
            total_loss += float(loss.item())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if is_main_process():
            if self.logger:
                self.logger.info(f"Validation epoch [{epoch + 1}] loss={avg_loss:.4f}")
            if self.tensorboard:
                self.tensorboard.add_scalar("val/loss", avg_loss, epoch + 1)
        if is_distributed():
            dist.barrier()
        return avg_loss

    def run(self):
        if self.logger:
            self.logger.info(
                f"Start PredRNN training: world_size={self.world_size}, "
                f"batch_size={self.batch_size}, total_length={self.total_length}, "
                f"img_size={self.img_size}, patch_size={self.patch_size}"
            )

        for epoch in range(self.epoch, self.max_epochs):
            train_loss = self.train_epoch(epoch)
            if is_main_process():
                if self.logger:
                    self.logger.info(
                        f"Epoch [{epoch + 1}/{self.max_epochs}] train_loss={train_loss:.4f}"
                    )
                process_checkpoint(
                    self.model,
                    self.optimizer,
                    self.workspace,
                    epoch,
                )

            if self.val_dataloader is not None and self.val_interval > 0:
                if (epoch + 1) % self.val_interval == 0:
                    self.validate(epoch)

        if is_main_process() and self.tensorboard:
            self.tensorboard.close()
