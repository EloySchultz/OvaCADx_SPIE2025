

#Please follow the following order for preprocessing data:
# 1. ovacadx/scripts/preprocess_data.py
# 2. Data_management/Create_faster_ovarian.py

# This file will:
# Open each nifti file
# Crop around the tumor region for both image and mask
# Save the adjusted niftis in a new folder (using the same folder structure as the original)
# Copy datasets_preprocessed.csv to the new folder


import pandas as pd
import os
from tqdm import tqdm
DATA_DIR =  "/home/eloy/Documents/Data/LUNG_DATA/NIFTI-Processed_new/"
df = pd.read_csv(os.path.join(DATA_DIR,"preprocessed_data.csv"))


import SimpleITK as sitk
import numpy as np
import os

def crop_and_pad_image(image_path, annot_path, target_folder, padding=2):
    # Load image and annotation using SimpleITK
    image = sitk.ReadImage(image_path)
    annotation = sitk.ReadImage(annot_path)

    # Get non-zero indices of the annotation
    annot_arr = sitk.GetArrayFromImage(annotation)
    annot_arr = np.transpose(annot_arr, (2, 1, 0)) #Put Z in the back again
    non_zero_indices = np.nonzero(annot_arr)

    # Get minimum and maximum indices along each axis
    min_indices = np.min(non_zero_indices, axis=1)
    max_indices = np.max(non_zero_indices, axis=1)

    # Calculate the bounding box dimensions
    start_x, start_y, start_z = min_indices
    end_x, end_y, end_z = max_indices

    # Crop the image and annotation

    image_size = image.GetSize()

    # Clip start indices
    start_x_clipped = max(0, start_x - padding)
    start_y_clipped = max(0, start_y - padding)
    start_z_clipped = max(0, start_z - padding)

    # Clip end indices
    end_x_clipped = min(image_size[0] - 1, end_x + padding)
    end_y_clipped = min(image_size[1] - 1, end_y + padding)
    end_z_clipped = min(image_size[2] - 1, end_z + padding)

    # Extract padded sub-volumes with clipped indices
    padded_image = image[start_x_clipped:end_x_clipped + 1, start_y_clipped:end_y_clipped + 1,
                   start_z_clipped:end_z_clipped + 1]
    padded_annotation = annotation[start_x_clipped:end_x_clipped + 1, start_y_clipped:end_y_clipped + 1,
                       start_z_clipped:end_z_clipped + 1]
    if 0 in list(padded_image.GetSize()):
        print("One of the dimensions of the padded anc clipped image is zero! Please check the dimensions")
        raise ValueError("One of the dimensions of the padded anc clipped image is zero! Please check the dimensions")

    # Create target folders if they don't exist
    os.makedirs(os.path.join(target_folder, "IMAGES"), exist_ok=True)
    os.makedirs(os.path.join(target_folder, "MASKS"), exist_ok=True)

    # Save the cropped and padded images
    sitk.WriteImage(padded_image, os.path.join(target_folder, "IMAGES", os.path.basename(image_path)))
    sitk.WriteImage(padded_annotation, os.path.join(target_folder, "MASKS", os.path.basename(annot_path)))
    print(os.path.join(target_folder, "IMAGES", os.path.basename(image_path)))

# Iterate over each row in the DataFrame
total_rows = len(df)
OUTPUT_FOLDER = os.path.join(DATA_DIR.rstrip(os.path.sep)+"_FAST")
new_image_paths = []
new_annot_paths = []
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# Iterate over each row in the DataFrame with tqdm progress bar
for index, row in tqdm(df.iterrows(), total=total_rows, desc="Processing Rows"):
    image_path = row['image_path']
    annot_path = row['annot_path']
    target_folder = os.path.join(OUTPUT_FOLDER, row['tumor_id']) # Set your target folder here
    crop_and_pad_image(image_path, annot_path, target_folder)
    new_image_paths.append(os.path.join(target_folder, "IMAGES", os.path.basename(image_path)))
    new_annot_paths.append(os.path.join(target_folder, "MASKS", os.path.basename(annot_path)))
# Update the DataFrame with new paths
df['image_path'] = new_image_paths
df['annot_path'] = new_annot_paths

df.to_csv((os.path.join(OUTPUT_FOLDER,"preprocessed_data.csv")),index=False)