import os
import argparse
from pathlib import Path

import pandas as pd
import SimpleITK as sitk


def get_args_parser():
    parser = argparse.ArgumentParser(description='Preprocess data for ovarian tumor classification')
    parser.add_argument('--data_path', type=str, default='/share/colon/cclaessens/datasets/Data_gyn_3.5/')  # default='/path/to/data/folder', help='Path to dataset folder')
    parser.add_argument('--output_path', type=str, default='/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/')  # default='/path/to/output/folder', help='Path to output folder')

    return parser


def resample_slice_distance(image, mask, slice_distance):
    """
    Resample image and mask to the slice distance
    """
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())

    resample.SetOutputSpacing((image.GetSpacing()[0], image.GetSpacing()[1], slice_distance))
    resample.SetSize((image.GetSize()[0], image.GetSize()[1], int(image.GetSpacing()[2] / slice_distance * image.GetSize()[2])))
    resampled_image = resample.Execute(image)

    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_mask = resample.Execute(mask)

    return resampled_image, resampled_mask


def main(args):

    # Read excel file containing the dataset information
    if os.path.isfile(os.path.join(args.data_path, "dataset.csv")):
        df = pd.read_csv(os.path.join(args.data_path, "dataset.csv"))
    else:
        raise FileNotFoundError("`preprocess_data.py` expects an excel file named `dataset.csv` in the dataset folder")
    # Select tumors that have `Inclusion` criteria
    if 'Inclusion' in df.columns:
        df = df[(df['Inclusion'] == 'Yes') | (df['Inclusion'] == 'yes') | (df['Inclusion'] == 1)]
    else:
        print("Column `Inclusion` not found in the dataset excel file, so all tumors will be processed")
    
    # Create dataframe instance to store information about the preprocessed images and masks
    preprocessed_df = pd.DataFrame(columns=[
        'patient_id',
        'tumor_id',
        'image_path',
        'annot_path',
        'label',
        'tumor_xstart_idx',
        'tumor_xend_idx',
        'tumor_ystart_idx',
        'tumor_yend_idx',
        'tumor_zstart_idx',
        'tumor_zend_idx',
        'is_internal_testset',
        'is_internal_testset_tiny',
        'is_external_testset',
    ])
    
    # Create label image statistics filter instance
    label_stats_filter = sitk.LabelStatisticsImageFilter()

    # Process images and masks for each tumor
    for idx, row in df.iterrows():

        # Get patient and tumor ID belonging to the tumor
        if 'Patient ID' in row and 'Tumor ID' in row:
            patient_id = row['Patient ID']
            tumor_id = row['Tumor ID']
        else:
            print(f"Patient and/or tumor ID not found")
            continue

        if 'Image path' in row and 'Annotation path' in row and os.path.isfile(
            os.path.join(args.data_path, row['Image path'])) and os.path.isfile(
                os.path.join(args.data_path, row['Annotation path'])):
            image_path = os.path.join(args.data_path, row['Image path'])
            mask_path = os.path.join(args.data_path, row['Annotation path'])
        else:
            print(f"Image and/or mask not found for tumor {tumor_id}")
            print(f"Image path: {os.path.join(args.data_path, row['Image path'])}")
            print(f"Mask path: {os.path.join(args.data_path, row['Annotation path'])}")
            continue
        
        # Read image and mask
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        if image.GetSize() != mask.GetSize():
            print(f"Image and mask sizes do not match for tumor {tumor_id}")
            continue

        if any(abs(img_space - msk_space) > 1e-6 for img_space, msk_space in zip(image.GetSpacing(), mask.GetSpacing())):
            print(f"Image and mask spacings do not match for tumor {tumor_id}, however, sizes do match --> resetting to match spacing of image")
            mask.SetSpacing(image.GetSpacing())

        # Resample image and mask to the slice distance of 3.0 mm
        if image.GetSpacing()[2] != 3.0:
            image, mask = resample_slice_distance(image, mask, 3.0)
        
        # Check if labels are present in the mask
        label_stats_filter.Execute(image, mask)
        if label_stats_filter.GetNumberOfLabels() == 0:
            print(f"No labels found in the mask for tumor {tumor_id}")
            continue

        # Get bounding box of the tumor
        bounding_box = label_stats_filter.GetBoundingBox(1)

        # Get classification labels of the tumors
        if 'Label' in row and row['Label'] in ['B', 'M', 'BL']:
            label = 0 if row['Label'] == 'B' else 1 if row['Label'] == 'M' else 2
        else:
            print(f"Label not found for tumor {tumor_id}, is expected to be one of ['B', 'M', 'BL']")
            continue
        
        assert 'Internal testset' in row and 'External testset' in row, "Internal and External testset columns are required"

        # Make directories to store preprocessed images and masks
        if row['Image path'].endswith('.nii'):
            row['Image path'] = row['Image path'].replace('.nii', '.nii.gz')
        if row['Annotation path'].endswith('.nii'):
            row['Annotation path'] = row['Annotation path'].replace('.nii', '.nii.gz')
        Path(os.path.join(args.output_path, Path(row['Image path']).parent)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.output_path, Path(row['Annotation path']).parent)).mkdir(parents=True, exist_ok=True)

        # Write image and mask to the output folder
        try:
            sitk.WriteImage(image, os.path.join(args.output_path, row['Image path']))
            sitk.WriteImage(mask, os.path.join(args.output_path, row['Annotation path']))
        except Exception as e:
            print(f"Error writing image and mask for tumor {tumor_id}")
            print(f"Error: {e}")
            continue
        # Append information to the preprocessed dataframe
        preprocessed_df = pd.concat([preprocessed_df, pd.DataFrame({
            'patient_id': patient_id,
            'tumor_id': tumor_id,
            'center': row['Center'],
            'image_path': os.path.join(args.output_path, row['Image path']),
            'annot_path': os.path.join(args.output_path, row['Annotation path']),
            'label': label,
            'tumor_xstart_idx': bounding_box[0],
            'tumor_xend_idx': bounding_box[1],
            'tumor_ystart_idx': bounding_box[2],
            'tumor_yend_idx': bounding_box[3],
            'tumor_zstart_idx': bounding_box[4],
            'tumor_zend_idx': bounding_box[5],
            'tumor_num_slices': bounding_box[5] - bounding_box[4],
            'is_internal_testset': 1 if (row['Internal testset'] == 'Yes') | (row['Internal testset'] == 'yes') | (row['Internal testset'] == 1) else 0,
            'is_internal_testset_tiny': 1 if (row['Internal testset tiny'] == 'Yes') | (row['Internal testset tiny'] == 'yes') | (row['Internal testset tiny'] == 1) else 0,
            'is_external_testset': 1 if (row['External testset'] == 'Yes') | (row['External testset'] == 'yes') | (row['External testset'] == 1) else 0,
            'outlier': 1 if (row['outlier'] == 1) else 0,

        }, index=[0])], ignore_index=True)
        print(f"Processed tumor {tumor_id}")

    # Sort values based on patient ID
    preprocessed_df = preprocessed_df.sort_values(by=['patient_id', 'tumor_id', 'label'], ignore_index=True)

    # Save the preprocessed dataframe to a csv file
    preprocessed_df.to_csv(os.path.join(args.output_path, "preprocessed_data.csv"), index=False)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
