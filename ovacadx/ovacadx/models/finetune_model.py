from copy import deepcopy
from typing import Optional, Callable, Union, List, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Metric, MetricCollection

from ovacadx.utils import get_optimizer_stats


class FineTuneModel(pl.LightningModule):
    def __init__(
            self, 
            model: nn.Module,
            batch_size: int,
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[LRScheduler] = None,
            criterion: Optional[Callable] = None,
            metrics: Union[Metric, List[Metric], Dict[str, Metric]] = None,
        ):
        super().__init__()
        self.model = model
        self.batch_size = batch_size

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_metrics = MetricCollection(metrics) if metrics is not None else None
        self.val_metrics = MetricCollection(deepcopy(metrics)) if metrics is not None else None
        self.test_metrics = MetricCollection(deepcopy(metrics)) if metrics is not None else None

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):

        inputs, targets = batch
        preds = self(inputs)

        loss = self.criterion(preds) if targets is None else self.criterion(preds, targets)
        self.log(
            "train/loss/epoch", 
            loss, 
            logger=True, 
            batch_size=self.batch_size, 
            on_epoch=True, 
            on_step=False,
        )

        if self.train_metrics is not None:
            self.train_metrics.update(preds, targets)

        return loss
    
    def on_train_epoch_end(self):
        if self.train_metrics is not None:
            for k, v in self.train_metrics.compute().items():
                self.log(
                    f"train/metrics/{k}/epoch", 
                    v, 
                    logger=True, 
                    batch_size=self.batch_size, 
                    on_epoch=True, 
                    on_step=False,
                    sync_dist=True,
                )
            self.train_metrics.reset()
        
        # for k, v in get_optimizer_stats(self.optimizer).items():
        #     self.log(f"train/{k}/epoch", v, logger=True, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
    
    def validation_step(self, batch, batch_idx):
        
        inputs, targets = batch
        preds = self(inputs)

        loss = self.criterion(preds) if targets is None else self.criterion(preds, targets)
        self.log(
            "val/loss/epoch", 
            loss, 
            logger=True, 
            batch_size=self.batch_size, 
            on_epoch=True, 
            on_step=False, 
        )

        if self.val_metrics is not None:
            self.val_metrics.update(preds, targets)
        
        return loss
    
    def on_validation_epoch_end(self):
        if self.val_metrics is not None:
            for k, v in self.val_metrics.compute().items():
                self.log(
                    f"val/metrics/{k}/epoch", 
                    v, 
                    logger=True, 
                    batch_size=self.batch_size,
                    on_epoch=True,
                    on_step=False, 
                    sync_dist=True,
                )
            self.val_metrics.reset()
    
    def test_step(self, batch, batch_idx):
            
            inputs, targets = batch
            preds = self(inputs)
    
            if self.test_metrics is not None:
                self.test_metrics.update(preds, targets)
    
    def on_test_epoch_end(self):
        if self.test_metrics is not None:
            for k, v in self.test_metrics.compute().items():
                self.log(
                    f"test/metrics/{k}/epoch", 
                    v, 
                    logger=True, 
                    batch_size=self.batch_size,
                    on_epoch=True,
                    on_step=False, 
                    sync_dist=True,
                )
            self.test_metrics.reset()
        

    def configure_optimizers(self):
        if self.optimizer is None:
            raise ValueError("Please specify what optimizer to use")
        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
