import os
import argparse
from glob import glob

import torch
import pandas as pd
import monai.transforms as transforms
import monai.networks.nets as monai_models

import ovacadx.models as models
from ovacadx.datasets import MILNestedKFoldModule
from ovacadx.utils import load_pretrained_weights_finetuned


def get_args_parser():
    parser = argparse.ArgumentParser('Evaluate fine-tuned models on internal and external testsets')
    parser.add_argument('--data_path', type=str, default='/path/to/dataset/csv', help='Path to dataset csv file')
    parser.add_argument('--label', type=str, default='label', help='Label column name in the dataset annotations')
    parser.add_argument('--orient', action='store_true', help='Whether to orient the images to LPI orientation')
    parser.add_argument('--mask_tumor', action='store_true', help='Whether to mask the tumor in the images')
    parser.add_argument('--crop_tumor', action='store_true', help='Whether to crop the tumor in the images')
    parser.add_argument('--key_external_testset', type=str, default='is_external_testset', help='Key for external testset')

    parser.add_argument('--backbone', type=str, default=None, help='Backbone classifier')
    parser.add_argument('--backbone_num_features', type=int, default=None, help='Number of output features of the backbone')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='Path to the pretrained weights')
    parser.add_argument('--mil_mode', type=str, default='att', choices=["mean", "max", "att", "att_trans", "att_trans_pyramid"], 
                        help='MIL algorithm')

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for data loading')

    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
    parser.add_argument('--num_inner_splits', type=int, default=5, help='Number of inner splits for nested cross-validation')
    parser.add_argument('--num_outer_splits', type=int, default=5, help='Number of outer splits for nested cross-validation')

    return parser


def main(args):
    transform = transforms.Compose([
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

    # Define MIL model
    model = models.MILModel(
        num_classes=1,
        mil_mode=args.mil_mode,
        backbone=backbone,
        backbone_num_features=num_features,
        pretrained=False,
    )
    
    ckpt_paths = sorted(glob(os.path.join(args.ckpt_dir, '*.ckpt')))

    k_folds = [int(os.path.basename(ckpt_path).split('-')[1].split('=')[1]) for ckpt_path in ckpt_paths]
    assert len(k_folds) == args.num_outer_splits * args.num_inner_splits

    columns = ['tumor_id'] + [f'output-k={k:02}' for k in k_folds]
    results_test = pd.DataFrame(columns=columns)
    results_val = pd.DataFrame(columns=columns)
    
    for k, ckpt_path in zip(k_folds, ckpt_paths):

        data_module = MILNestedKFoldModule(
            path=args.data_path, 
            label=args.label, 
            orient=args.orient, 
            mask_tumor=args.mask_tumor, 
            crop_tumor=args.crop_tumor,
            val_transform=transform,
            test_transform=transform,
            num_instances=25,
            key_external_testset=args.key_external_testset,
            k=k,  # fold number
            num_inner_splits=args.num_inner_splits,
            num_outer_splits=args.num_outer_splits,
            seed=args.seed,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            remove_outliers=False,
        )

        data_module.setup("fit")
        data_loader = data_module.test_dataloader()

        tumor_ids = data_module.test_samples

        model = load_pretrained_weights_finetuned(model, ckpt_path)
        model.eval()

        for tumor_id, data in zip(tumor_ids, data_loader):
            image, label = data
            image = image
            label = label
            with torch.no_grad():
                output = model(image)
                result = {'tumor_id': tumor_id, f'output-k={k:02}': output.item()}
            if result['tumor_id'] not in results_test['tumor_id'].values:
                results_test = results_test.append(result, ignore_index=True)
            else:
                results_test.loc[results_test['tumor_id'] == result['tumor_id'], f'output-k={k:02}'] = result[f'output-k={k:02}']
        
        data_loader = data_module.val_dataloader()

        tumor_ids = data_module.val_samples

        for tumor_id, data in zip(tumor_ids, data_loader):
            image, label = data
            image = image
            label = label
            with torch.no_grad():
                output = model(image)
                result = {'tumor_id': tumor_id, f'output-k={k:02}': output.item()}
            if result['tumor_id'] not in results_val['tumor_id'].values:
                results_val = results_val.append(result, ignore_index=True)
            else:
                results_val.loc[results_val['tumor_id'] == result['tumor_id'], f'output-k={k:02}'] = result[f'output-k={k:02}']

    results_test.to_csv(os.path.join(args.ckpt_dir, 'results_test.csv'), index=False)
    results_val.to_csv(os.path.join(args.ckpt_dir, 'results_val.csv'), index=False)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)