import pandas as pd
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm  # Progress bar library

#This fujnction can be used, but is kinda redundant since radiomics also has a volume feature. Still, this was used to generate the SPIE publication results.
def calculate_volume(annot_path, spacing):
    """
    Calculate the tumor volume (in cm³) from the annotation NIfTI file.

    Parameters:
    - annot_path (str): Path to the NIfTI file.
    - spacing (tuple): Voxel spacing (dx, dy, dz) in mm.

    Returns:
    - volume_cm3 (float): Tumor volume in cubic centimeters.
    """
    # Load the NIfTI file
    nii = nib.load(annot_path)
    data = nii.get_fdata()

    # Count the number of voxels where the tumor is present (value = 1)
    tumor_voxels = np.sum(data == 1)

    if tumor_voxels == 0:
        # No tumor detected
        return 0.0

    # Calculate the volume of a single voxel in mm³
    voxel_volume_mm3 = np.prod(spacing)

    # Calculate the total volume in mm³, then convert to cm³
    volume_cm3 = (tumor_voxels * voxel_volume_mm3) / 1000  # Convert mm³ to cm³
    return volume_cm3


def main():
    # Define the paths to the data directories
    paths = [
        "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new",
        "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new_FAST",
        "/home/eloy/Documents/Data/LUNG_DATA/NIFTI-Processed_new",
        "/home/eloy/Documents/Data/LUNG_DATA/NIFTI-Processed_new_FAST"
    ]

    for path in paths:
        print('Processing:' + str(path))

        if os.path.exists(os.path.join(path,"preprocesssed_data_original.csv")):
              print("Skipped as sizes were already calculated")
              continue;

        # Load the CSV file into a dataframe
        original_file = os.path.join(path, 'preprocessed_data.csv')
        if not os.path.exists(original_file):
            print(f"File not found: {original_file}")
            continue

        df = pd.read_csv(original_file)

        if "volume_cm3" in df.keys():
            print("Skipped as sizes were already calculated")
            continue;

        # Save the original dataframe as preprocessed_data_original.csv
        original_backup_file = os.path.join(path, 'preprocessed_data_original.csv')
        df.to_csv(original_backup_file, index=False)
        print(f"Original dataframe saved to {original_backup_file}")

        # List to store tumor volumes
        volumes = []

        # Iterate through each row in the dataframe with a progress bar
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {path}"):
            annot_path = row['annot_path']

            try:
                # Load NIfTI file to get voxel spacing
                nii = nib.load(annot_path)
                spacing = nii.header.get_zooms()  # Spacing in mm

                # Calculate the tumor volume
                volume_cm3 = calculate_volume(annot_path, spacing)
            except Exception as e:
                print(f"Error processing {annot_path}: {e}")
                volume_cm3 = 0.0

            # Append the result to the list
            volumes.append(volume_cm3)

        # Add the volume column to the dataframe
        df['volume_cm3'] = volumes

        # Save the updated dataframe as preprocessed_data.csv
        updated_file = os.path.join(path, 'preprocessed_data.csv')
        df.to_csv(updated_file, index=False)
        print(f"Updated dataframe saved to {updated_file}")


if __name__ == "__main__":
    main()
