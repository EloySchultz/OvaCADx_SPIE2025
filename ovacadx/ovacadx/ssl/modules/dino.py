# Mostly copy paste from https://docs.lightly.ai/self-supervised-learning/examples/dino.html
from copy import deepcopy

import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from timm.optim import Lars
from torchmetrics import MetricCollection

from ovacadx.schedulers import LinearWarmupCosineAnnealingLR
from ovacadx.utils import get_optimizer_stats


class DINO(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            num_ftrs: int,
            hidden_dim: int = 2048,
            bottleneck_dim: int = 256,
            output_dim: int = 65536,
            momentum_teacher: float = 0.996,
            warmup_teacher_temp: float = 0.04,
            teacher_temp: float = 0.04,
            warmup_teacher_temp_epochs: int = 0,
            student_temp: float = 0.1,
            center_momentum: float = 0.9,
            optimizer: str = 'adamw',
            lr: float = 0.0005,
            min_lr: float = 1e-6,
            warmup_epochs: int = 10,
            batch_size: int = 32,
            steps_per_epoch: int = 0,
        ):
        super().__init__()

        self.num_ftrs = num_ftrs
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.momentum_teacher = momentum_teacher
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.optimizer = optimizer
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            num_ftrs, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=1
        )
        self.teacher_backbone = deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(num_ftrs, hidden_dim, bottleneck_dim, output_dim)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(
            output_dim=output_dim, 
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            student_temp=student_temp,
            center_momentum=center_momentum,
        )

    def forward(self, x):
        y = self.student_backbone(x)
        if isinstance(y, tuple):
            y = y[0]
        z = self.student_head(y.flatten(start_dim=1))
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x)
        if isinstance(y, tuple):
            y = y[0]
        z = self.teacher_head(y.flatten(start_dim=1))
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(
            self.global_step, 
            self.steps_per_epoch * self.trainer.max_epochs, 
            self.momentum_teacher, 
            1,
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = [view.to(self.device) for view in batch]  # have to specifically do this here
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        self._log_stats(loss, None, "train", batch_idx)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        # linear scaling rule for learning rate
        lr = self.lr * (self.batch_size * self.trainer.world_size) / 256.
        # optimizer
        if self.optimizer == 'adamw':
            optim = torch.optim.AdamW(self.parameters(), lr=lr)
        elif self.optimizer == 'sgd':
            optim = torch.optim.SGD(self.parameters(), lr=lr, moentum=0.9)  # lr will be set by scheduler
        elif self.optimizer == 'lars':
            optim = Lars(self.parameters(), lr=lr)
        else:
            raise ValueError(f'Unsupported optimizer: {self.optimizer}')
        
        # scheduler
        scheduler = LinearWarmupCosineAnnealingLR(
            optim,
            warmup_epochs=self.steps_per_epoch * self.warmup_epochs,
            max_epochs=self.steps_per_epoch * self.trainer.max_epochs,
            eta_min=self.min_lr,
        )
        # learning rate scheduler
        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optim], [lr_scheduler]
    
    def _log_stats(self, loss: torch.Tensor, metrics: MetricCollection, mode: str, batch_idx: int) -> None:
        if self.trainer.logger is None:
            return

        # Arguments for self.log()
        log_kwargs = {"logger": True, "batch_size": self.batch_size}
        on_step_log_kwargs = {"on_epoch": False, "on_step": True, "sync_dist": False}
        on_epoch_log_kwargs = {"on_epoch": True, "on_step": False, "sync_dist": True}

        # Loss
        if loss is not None:
            self.log(f"{mode}/loss/step", loss, **log_kwargs, **on_step_log_kwargs)
            self.log(f"{mode}/loss/epoch", loss, **log_kwargs, **on_epoch_log_kwargs)
        # Metrics
        if metrics is not None:
            for k, v in metrics.items():
                self.log(f"{mode}/metrics/{k}/step", v, **log_kwargs, **on_step_log_kwargs)
                self.log(f"{mode}/metrics/{k}/epoch", v, **log_kwargs, **on_epoch_log_kwargs)
        # Optimizer's lr, momentum, beta. Logged in train mode and once per epoch.
        if mode == "train":  # and batch_idx == 0:
            for k, v in get_optimizer_stats(self.trainer.optimizers[0]).items():
                self.log(f"{mode}/{k}/step", v, **log_kwargs, **on_step_log_kwargs)
                self.log(f"{mode}/{k}/epoch", v, **log_kwargs, **on_epoch_log_kwargs)
