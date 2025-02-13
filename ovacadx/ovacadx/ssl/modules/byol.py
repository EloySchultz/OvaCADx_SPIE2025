# Mostly copy paste from https://docs.lightly.ai/self-supervised-learning/examples/dino.html
from copy import deepcopy

import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead
from lightly.models.modules.heads import BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from timm.optim import Lars
from torchmetrics import MetricCollection

from ovacadx.schedulers import LinearWarmupCosineAnnealingLR
from ovacadx.utils import get_optimizer_stats


class BYOL(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            backbone_momentum: nn.Module,  # necessary for forward hooks which don't work when deepcopying the backbone
            num_ftrs: int,
            hidden_dim: int = 4096,
            output_dim: int = 256,
            momentum_teacher: float = 0.996,
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
        self.output_dim = output_dim
        self.momentum_teacher = momentum_teacher
        self.optimizer = optimizer
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(num_ftrs, hidden_dim, output_dim)
        self.prediction_head = BYOLPredictionHead(output_dim, hidden_dim, output_dim)

        self.backbone_momentum = backbone_momentum
        self.projection_head_momentum = deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        y = self.backbone(x)
        z = self.projection_head(y.flatten(start_dim=1))
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x)
        z = self.projection_head_momentum(y.flatten(start_dim=1))
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(
            self.global_step, 
            self.steps_per_epoch * self.trainer.max_epochs, 
            self.momentum_teacher, 
            1,
        )
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        (x0, x1) = batch

        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))

        self._log_stats(loss, None, "train", batch_idx)
        return loss

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
