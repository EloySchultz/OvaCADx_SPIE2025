import os
import math
import argparse
from glob import glob

from torch.utils.data.dataloader import default_collate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from monai.utils import set_determinism
import monai.transforms as transforms
import monai.data as data

import ovacadx.ssl.models as ssl_models
from ovacadx.ssl.modules import DINO
from ovacadx.transforms import MultiCrop


# Argument parser
def get_args_parser():
    parser = argparse.ArgumentParser(description='Pretrain DINO model on abdomen CT slices')
    parser.add_argument('--data_path', type=str, default='/path/to/dataset/', help='Path to dataset folder')

    parser.add_argument('--backbone', type=str, default=None, help='Backbone architecture')
    parser.add_argument('--hidden_dim', type=int, default=2048, help='Hidden dimension of the projection head')
    parser.add_argument('--bottleneck_dim', type=int, default=256, help='Bottleneck dimension of the projection head')
    parser.add_argument('--output_dim', type=int, default=65536, help='Output dimension of the projection head')
    parser.add_argument('--momentum_teacher', type=float, default=0.996, help='Momentum for updating teacher network')
    parser.add_argument('--warmup_teacher_temp', type=float, default=0.04, help='Warmup temperature for teacher network')
    parser.add_argument('--teacher_temp', type=float, default=0.04, help='Temperature for teacher network')
    parser.add_argument('--warmup_teacher_temp_epochs', type=int, default=0, help='Number of epochs for warmup temperature')
    parser.add_argument('--student_temp', type=float, default=0.1, help='Temperature for student network')
    parser.add_argument('--center_momentum', type=float, default=0.9, help='Momentum for updating center')
    
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd', 'lars'], help='Optimizer')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training')
    parser.add_argument('--drop_last_batch', action='store_true', help='Drop the last batch if smaller than batch size')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for data loading')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
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
        transforms.Orientation('RA'),
        transforms.ScaleIntensityRange(
            a_min=-100, 
            a_max=300, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True,
        ),
        transforms.Resize(spatial_size=[512, 512]),
        MultiCrop(
            high_resolution_transforms=transforms.Compose([
                transforms.RandSpatialCropSamples(
                    roi_size=[256, 256],
                    max_roi_size=[512, 512],
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
                transforms.Resize(spatial_size=[384, 384]),
            ]),
            low_resolution_transforms=transforms.Compose([
                transforms.RandSpatialCropSamples(
                    roi_size=[64, 64],
                    max_roi_size=[256, 256],
                    random_center=True,
                    random_size=True,
                    num_samples=8,
                ),
                transforms.RandRotate(prob=0.5, range_x=math.pi / 6),
                transforms.RandAxisFlip(prob=0.5),
                transforms.OneOf([
                    transforms.RandGaussianSharpen(prob=0.5),
                    transforms.RandGaussianSmooth(prob=0.5),
                ]),
                transforms.Resize(spatial_size=[128, 128]),
            ]),
        )
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

    # Define MIL model
    dino_module = DINO(
        backbone=backbone,
        num_ftrs=num_ftrs,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        output_dim=args.output_dim,
        momentum_teacher=args.momentum_teacher,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
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
        logger=WandbLogger(name=f'pretrain_slice_encoder_{args.backbone}', project='ovacadx'),
        log_every_n_steps=20,
        callbacks=[
            ModelCheckpoint(
                dirpath=f'checkpoints/pretraining/slice_encoder/{args.backbone}', 
                filename='model-{epoch:03d}-{val_loss:.2f}'
            )
        ],
        num_nodes=args.nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=True,
        # deterministic=True,
    )

    # Train the model
    trainer.fit(dino_module, data_loader)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
