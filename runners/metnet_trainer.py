import os
import random
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.builder import (
    build_criterion,
    build_dataloader,
    build_model,
    build_optimizer,
    build_scheduler,
)
from utils.checkpoint import process_checkpoint, restore_model
from utils.logger import get_logger
from utils.misc import is_distributed, is_main_process
from utils.registry import register_module


@register_module(parent="runners")
class MetNetTrainer:
    def __init__(
        self,
        dataset: dict,
        model: dict,
        criterion: dict,
        workspace: str,
        batch_size: int = 1,
        num_workers: int = 4,
        max_epochs: int = 10,
        val_interval: int = 1,
        enable_amp: bool = True,
        forecast_steps_train: int = 4,
        forecast_steps_val: Optional[int] = None,
        log_interval: int = 10,
        pretrained_weight_path: Optional[str] = None,
        resume: bool = False,
        find_unused_parameters: bool = False,
        optimizer: Optional[dict] = None,
        scheduler: Optional[dict] = None,
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
        self.logger = get_logger("metnet", log_file=log_file) if is_main_process() else None
        self.tensorboard = SummaryWriter(self.workspace) if is_main_process() else None

        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.enable_amp = enable_amp and torch.cuda.is_available()
        self.forecast_steps_train = forecast_steps_train
        self.forecast_steps_val = forecast_steps_val or forecast_steps_train
        self.log_interval = log_interval
        self.global_step = 0
        self.epoch = 0

        dataset_cfg = dataset
        model_cfg = model
        criterion_cfg = criterion

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

        self.model_cfg = model_cfg
        self.forecast_steps = model_cfg.get("forecast_steps", 96)
        self.model = build_model(
            model_cfg,
            device=self.device,
            logger=self.logger,
            pretrained_weight_path=None if resume else pretrained_weight_path,
            find_unused_parameters=find_unused_parameters,
        )
        self.criterion = build_criterion(criterion_cfg)
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

        if resume and pretrained_weight_path is not None:
            checkpoint = torch.load(pretrained_weight_path, map_location="cpu")
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "epoch" in checkpoint:
                self.epoch = checkpoint["epoch"]
            if self.logger:
                self.logger.info(f"Resumed from epoch {self.epoch}")

    def _set_train_epoch(self, epoch: int):
        if is_distributed():
            self.train_dataloader.sampler.set_epoch(epoch)

    def _unwrap_model(self):
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        return model

    def _sample_lead_times(self, count: int):
        count = min(count, self.forecast_steps)
        return random.sample(range(self.forecast_steps), k=count)

    def _forward_lead_times(self, inputs, targets, lead_times, backward: bool):
        total_loss = 0.0
        loss_dict = {}
        num_lead_times = len(lead_times)

        if backward:
            self.optimizer.zero_grad(set_to_none=True)

        for lead_time in lead_times:
            target_t = targets[:, lead_time]
            with torch.cuda.amp.autocast(enabled=self.enable_amp):
                pred = self.model(inputs, lead_time=lead_time)
                loss = self.criterion(pred, target_t) / num_lead_times

            if backward:
                self.scaler.scale(loss).backward()
            total_loss += loss.item()
            loss_dict[f"lead_{lead_time}"] = loss.item()

        if backward:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

        loss_dict["total"] = total_loss
        return total_loss, loss_dict

    def train_epoch(self, epoch: int):
        self.model.train()
        self._set_train_epoch(epoch)
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_dataloader):
            inputs = batch["inputs"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)
            lead_times = self._sample_lead_times(self.forecast_steps_train)

            loss, loss_dict = self._forward_lead_times(
                inputs, targets, lead_times, backward=True
            )
            epoch_loss += loss
            num_batches += 1
            self.global_step += 1

            if is_main_process() and (batch_idx + 1) % self.log_interval == 0:
                elapsed = time.time() - start_time
                if self.logger:
                    self.logger.info(
                        f"Epoch [{epoch + 1}/{self.max_epochs}] "
                        f"Iter [{batch_idx + 1}/{len(self.train_dataloader)}] "
                        f"loss={loss:.4f} time={elapsed:.1f}s"
                    )
                if self.tensorboard:
                    self.tensorboard.add_scalar("train/loss", loss, self.global_step)

        return epoch_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, epoch: int):
        if self.val_dataloader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        lead_times = list(range(min(self.forecast_steps_val, self.forecast_steps)))

        for batch in self.val_dataloader:
            inputs = batch["inputs"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)
            batch_loss = 0.0
            for lead_time in lead_times:
                target_t = targets[:, lead_time]
                with torch.cuda.amp.autocast(enabled=self.enable_amp):
                    pred = self.model(inputs, lead_time=lead_time)
                    batch_loss += self.criterion(pred, target_t).item()
            total_loss += batch_loss / len(lead_times)
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
                f"Start training: world_size={self.world_size}, "
                f"batch_size={self.batch_size}, forecast_steps_train={self.forecast_steps_train}"
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
