import os
import math
import argparse
from glob import glob
from copy import deepcopy

import torch.nn as nn
from torch.utils.data.dataloader import default_collate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import monai.transforms as transforms
import monai.data as data
from monai.utils import set_determinism

import ovacadx.models as models
import ovacadx.ssl.models as ssl_models
from ovacadx.ssl.modules import BYOL
from ovacadx.transforms import Einsum
from ovacadx.utils import load_pretrained_weights_dino


# Argument parser
def get_args_parser():
    parser = argparse.ArgumentParser(description='Pretrain DINO model on abdomen CT slices')
    parser.add_argument('--data_path', type=str, default='/path/to/dataset/', help='Path to dataset folder')
    parser.add_argument('--mil_mode', type=str, default='att_trans', choices=["att_trans", "att_trans_pyramid"], 
                        help='MIL algorithm')

    parser.add_argument('--backbone', type=str, default=None, help='Backbone architecture')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to the pretrained weights')
    parser.add_argument('--hidden_dim', type=int, default=4096, help='Hidden dimension of the projection head')
    parser.add_argument('--output_dim', type=int, default=256, help='Output dimension of the projection head')
    parser.add_argument('--momentum_teacher', type=float, default=0.996, help='Momentum for updating teacher network')

    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd', 'lars'], help='Optimizer')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training')
    parser.add_argument('--drop_last_batch', action='store_true', help='Drop the last batch if smaller than batch size')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for data loading')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs per node')
    
    return parser


def main(args):
    # Set seed for reproducibility
    pl.seed_everything(args.seed)
    set_determinism(seed=args.seed)

    # Define transforms
    transform = transforms.Compose([
        transforms.LoadImage(),
        transforms.EnsureChannelFirst(channel_dim="no_channel"),
        transforms.Orientation('RAS'),
        transforms.ScaleIntensityRange(
            a_min=-100, 
            a_max=300, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True,
        ),
        transforms.Resize(spatial_size=[512, 512, -1]),
        transforms.RandSpatialCrop(roi_size=[512, 512, 64]),
        transforms.RandSpatialCropSamples(
            roi_size=[64, 64, 8],
            max_roi_size=[512, 512, 64],
            random_center=True,
            random_size=True,
            num_samples=2,
        ),
        transforms.RandRotate(prob=0.5, range_x=math.pi / 6),
        transforms.RandAxisFlip(prob=0.5),
        transforms.OneOf([
            transforms.RandGaussianSharpen(prob=0.5),
            transforms.RandGaussianSmooth(prob=0.5),
        ]),
        transforms.Resize(spatial_size=[384, 384, 24]),
        Einsum(equation="chwd->dchw")
    ])

    # Load dataset
    image_paths = glob(os.path.join(args.data_path, '*.nii.gz'))
    print(f'Found {len(image_paths)} images in {args.data_path}')
    dataset = data.Dataset(
        data=image_paths,
        transform=transform,
    )

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=default_collate,
        drop_last=args.drop_last_batch,
    )
    
    # Load backbone
    if args.backbone in ssl_models.__dict__ and args.backbone.startswith('vit'):
        backbone = ssl_models.__dict__[args.backbone](
            in_channels=1,
            spatial_dims=2,
            img_size=384,
            classification=True,
        )
        num_ftrs = backbone.embed_dim
    elif args.backbone in ssl_models.__dict__ and args.backbone.startswith('resnet'):
        backbone = ssl_models.__dict__[args.backbone](
            n_input_channels=1,
            spatial_dims=2,
            feed_forward=True,
        )
        num_ftrs = backbone.fc.in_features
        backbone.fc = None
    else:
        raise ValueError(f'Backbone {args.backbone} not supported')
    
    if args.pretrained_path is not None:
        backbone = load_pretrained_weights_dino(backbone, args.pretrained_path)
        backbone.requires_grad_(False)
    
    model = models.MILModel(
        num_classes=1,
        mil_mode=args.mil_mode,
        backbone=backbone,
        backbone_num_features=num_ftrs,
    )
    model.myfc = nn.Identity()

    model_momentum = models.MILModel(
        num_classes=1,
        mil_mode=args.mil_mode,
        backbone=deepcopy(backbone),
        backbone_num_features=num_ftrs,
    )
    num_ftrs = model_momentum.myfc.in_features
    model_momentum.myfc = nn.Identity()

    # Define MIL model
    byol_module = BYOL(
        backbone=model,
        backbone_momentum=model_momentum,
        num_ftrs=num_ftrs,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        momentum_teacher=args.momentum_teacher,
        optimizer=args.optimizer,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        steps_per_epoch=len(data_loader) // (args.gpus * args.nodes),
    )

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=WandbLogger(name=f'pretrain_volume_encoder_{args.mil_mode}_{args.backbone}', project='ovacadx'),
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=f'checkpoints/pretraining/volume_encoder/{args.mil_mode}/{args.backbone}', 
                filename='model-{epoch:03d}-{val_loss:.2f}'
            )
        ],
        num_nodes=args.nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=True,
        deterministic=True,
    )

    # Train the model
    trainer.fit(byol_module, data_loader)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
