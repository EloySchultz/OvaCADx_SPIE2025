
import math
import argparse

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch import seed_everything
import monai.transforms as transforms
import monai.networks.nets as monai_models
from monai.utils import set_determinism

import ovacadx.models as models
from ovacadx.utils import load_pretrained_weights_byol, load_pretrained_weights_dino
from ovacadx.datasets import MILNestedKFoldModule


# Argument parser
def get_args_parser():
    parser = argparse.ArgumentParser(description='Fine-tune backbone and MIL model for ovarian tumor classification')
    parser.add_argument('--data_path', type=str, default='/path/to/dataset/csv', help='Path to dataset csv file')
    parser.add_argument('--label', type=str, default='label', help='Label column name in the dataset annotations')
    parser.add_argument('--orient', action='store_true', help='Whether to orient the images to LPI orientation')
    parser.add_argument('--mask_tumor', action='store_true', help='Whether to mask the tumor in the images')
    parser.add_argument('--crop_tumor', action='store_true', help='Whether to crop the tumor in the images')

    parser.add_argument('--backbone', type=str, default=None, help='Backbone classifier')
    parser.add_argument('--backbone_num_features', type=int, default=None, help='Number of output features of the backbone')
    parser.add_argument('--pretrained', action='store_true', help='Init backbone with pretrained weights')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to the pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone weights')
    parser.add_argument('--mil_mode', type=str, default='att', choices=["mean", "max", "att", "att_trans", "att_trans_pyramid"], 
                        help='MIL algorithm')
    
    parser.add_argument('--do_rotate', action='store_true', help='Whether to apply random rotation')
    parser.add_argument('--do_translate', action='store_true', help='Whether to apply random translation')
    parser.add_argument('--do_scale', action='store_true', help='Whether to apply random scaling')
    
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'], help='Learning rate scheduler')
    parser.add_argument('--metrics', type=str, nargs='+', default=['Accuracy', 'AUROC'], help='Metrics for evaluation')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--drop_last_batch', action='store_true', help='Drop the last batch if smaller than batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for data loading')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
    parser.add_argument('--num_inner_splits', type=int, default=5, help='Number of inner splits for nested cross-validation')
    parser.add_argument('--num_outer_splits', type=int, default=5, help='Number of outer splits for nested cross-validation')
    parser.add_argument('--k', type=int, default=0, help='Fold number')
    
    return parser


def main(args):
    # Define transforms
    train_transform = transforms.Compose([
        transforms.EnsureChannelFirst(channel_dim="no_channel"),
        transforms.ScaleIntensityRange(
            a_min=-100, 
            a_max=300, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True,
        ),
        transforms.RandAffine(
            prob=0.5, 
            rotate_range=(0, 0, math.pi / 6) if args.do_rotate else None, 
            translate_range=(10, 10, 0) if args.do_translate else None, 
            scale_range=(0.1, 0.1, 0) if args.do_scale else None, 
            mode='trilinear',
        ),
        transforms.Resize((-1, 224, 224)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.EnsureChannelFirst(channel_dim="no_channel"),
        transforms.ScaleIntensityRange(
            a_min=-100, 
            a_max=300, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True,
        ),
        transforms.Resize((-1, 224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset
    data_module = MILNestedKFoldModule(
        path=args.data_path,
        label=args.label,
        orient=args.orient,
        mask_tumor=args.mask_tumor,
        crop_tumor=args.crop_tumor,
        train_transform=train_transform,
        val_transform=val_transform,
        num_instances=25,
        k=args.k,  # fold number
        num_inner_splits=args.num_inner_splits,
        num_outer_splits=args.num_outer_splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        seed=args.seed,
        remove_outliers=False,
        # key_internal_testset='is_internal_testset_tiny',
    )

    # Load backbone
    if args.backbone in monai_models.__dict__ and args.backbone.startswith('resnet'):
        backbone = monai_models.__dict__[args.backbone](
            n_input_channels=1,
            spatial_dims=2,
            num_classes=1,
            feed_forward=True,
        )
        num_features = backbone.fc.in_features
        backbone.fc = None
    else:
        backbone = args.backbone
    
    if not args.pretrained and args.pretrained_path is not None:
        backbone = load_pretrained_weights_dino(backbone, args.pretrained_path)
        backbone.requires_grad_(False)

    # Define MIL model
    model = models.MILModel(
        num_classes=1,
        mil_mode=args.mil_mode,
        backbone=backbone,
        backbone_num_features=num_features,
        pretrained=args.pretrained,
    )

    if args.pretrained and args.pretrained_path is not None:
        model = load_pretrained_weights_byol(model, args.pretrained_path)
        model.net.requires_grad_(False)


    # Define optimizer
    if args.freeze_backbone:
        for param in model.net.parameters():
            param.requires_grad = False
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise ValueError(f'Unsupported optimizer: {args.optimizer}')
    
    # Define learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    else:
        scheduler = None
    
    # Define loss function
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.55]))  # 1.65

    # Define metrics
    metrics = []
    for metric_name in args.metrics:
        if metric_name in torchmetrics.__dict__:
            metric = torchmetrics.__dict__[metric_name](task="binary")
        else:
            raise ValueError(f'Unsupported metric: {metric_name}')
        metrics.append(metric)
    
    # Initialize Lightning module
    finetune_module = models.FineTuneModel(
        model=model,
        batch_size=args.batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        metrics=metrics,
    )

    # Initialize PyTorch Lightning trainer
    filename_start = f'model-k={args.k:02d}-'  # for cross-validation
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=WandbLogger(name=f'finetune_{args.mil_mode}_{args.backbone}', project='ovacadx'),
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=f'checkpoints/downstream/{args.mil_mode}/{args.backbone}',
                monitor='val/metrics/BinaryAUROC/epoch',
                mode='max',
                filename=filename_start + 'epoch={epoch:03d}-val_loss={val/loss/epoch:.3f}-val_auc={val/metrics/BinaryAUROC/epoch:.3f}',
                auto_insert_metric_name=False,
            )
        ],
        deterministic=True,
    )

    # Train the model
    trainer.fit(finetune_module, data_module)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    seed_everything(args.seed)
    set_determinism(seed=args.seed)

    main(args)
