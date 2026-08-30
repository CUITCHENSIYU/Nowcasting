import os
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.builder import build_dataloader, build_model, build_optimizer, build_scheduler
from utils.checkpoint import process_checkpoint
from utils.logger import get_logger
from utils.misc import is_distributed, is_main_process
from utils.registry import register_module


@register_module(parent="runners")
class SkillfullTrainer:
    """Supervised trainer for Skillful Nowcasting (generator + evolution MSE).

    Full adversarial (GAN) training is not included yet; this matches the
    PredRNN-style single-optimizer loop for getting a runnable baseline.
    """

    def __init__(
        self,
        dataset: dict,
        model: dict,
        workspace: str,
        criterion: Optional[dict] = None,
        batch_size: int = 2,
        num_workers: int = 4,
        max_epochs: int = 50,
        val_interval: int = 5,
        enable_amp: bool = False,
        log_interval: int = 20,
        with_mask: bool = False,
        pretrained_weight_path: Optional[str] = None,
        resume: bool = False,
        find_unused_parameters: bool = False,
        optimizer: Optional[dict] = None,
        scheduler: Optional[dict] = None,
        evo_loss_weight: float = 0.5,
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
        self.logger = (
            get_logger("skillfull", log_file=log_file) if is_main_process() else None
        )
        self.tensorboard = SummaryWriter(self.workspace) if is_main_process() else None

        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.enable_amp = enable_amp and torch.cuda.is_available()
        self.log_interval = log_interval
        self.with_mask = with_mask
        self.evo_loss_weight = float(evo_loss_weight)
        self.global_step = 0
        self.epoch = 0

        dataset_cfg = dataset
        model_cfg = model.copy()

        self.seq_len = int(dataset_cfg["seq_len"])
        self.forecast_steps = int(dataset_cfg["forecast_steps"])
        self.total_length = self.seq_len + self.forecast_steps
        img_size = int(dataset_cfg.get("input_size", 256))

        model_cfg.setdefault("input_length", self.seq_len)
        model_cfg.setdefault("pred_length", self.forecast_steps)
        model_cfg.setdefault("total_length", self.total_length)
        model_cfg.setdefault("img_height", img_size)
        model_cfg.setdefault("img_width", img_size)

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

        # NowcastNet-style generator often has parameters not used in every forward
        # (spectral-norm hooks, dual evo/gen branches). DDP needs this enabled.
        ddp_find_unused = find_unused_parameters
        if is_distributed() and not find_unused_parameters:
            ddp_find_unused = True
            if self.logger:
                self.logger.info(
                    "DDP: auto-enabled find_unused_parameters=True for SkillfulNowcasting"
                )

        self.model = build_model(
            model_cfg,
            device=self.device,
            logger=self.logger,
            pretrained_weight_path=pretrained_weight_path,
            find_unused_parameters=ddp_find_unused,
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

    def _mse_loss(self, pred: torch.Tensor, target: torch.Tensor, with_mask: bool = False) -> torch.Tensor:
        """MSE averaged only over target pixels where value != 0 when with_mask=True."""
        sq_err = (pred - target).pow(2)
        if not with_mask:
            return sq_err.mean()
        mask = target.ne(0)
        if not mask.any():
            return sq_err.mean()
        return sq_err.masked_select(mask).mean()

    def _compute_loss(self, frames: torch.Tensor, with_mask: bool = False):
        gen_pred, evo_pred = self.model(frames)
        target = frames[:, self.seq_len : self.seq_len + self.forecast_steps]
        gen_loss = self._mse_loss(gen_pred, target, with_mask)
        evo_loss = self._mse_loss(evo_pred, target, with_mask)
        loss = gen_loss + self.evo_loss_weight * evo_loss
        return loss, float(gen_loss.detach()), float(evo_loss.detach())

    def train_epoch(self, epoch: int):
        self.model.train()
        self._set_train_epoch(epoch)
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_dataloader):
            frames = batch["frames"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.enable_amp):
                loss, gen_loss, evo_loss = self._compute_loss(frames, self.with_mask)

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
                        f"loss={loss_val:.6f} gen={gen_loss:.6f} evo={evo_loss:.6f} "
                        f"time={elapsed:.1f}s"
                    )
                if self.tensorboard:
                    self.tensorboard.add_scalar("train/loss", loss_val, self.global_step)
                    self.tensorboard.add_scalar("train/gen_loss", gen_loss, self.global_step)
                    self.tensorboard.add_scalar("train/evo_loss", evo_loss, self.global_step)

            del loss

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
            with torch.cuda.amp.autocast(enabled=self.enable_amp):
                loss, _, _ = self._compute_loss(frames, self.with_mask)
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
                f"Start Skillful Nowcasting training: world_size={self.world_size}, "
                f"batch_size={self.batch_size}, total_length={self.total_length}, "
                f"evo_loss_weight={self.evo_loss_weight}"
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
