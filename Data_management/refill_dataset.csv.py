import os
import pandas as pd
import re


#This file was used to check wether we miss any files in our dataset.csv and then add them accordingly.

#
# Final Program Overview:
# The program processes and validates a dataset of medical images (NIFTI files) and their corresponding annotations by grouping, checking, and updating the file paths in a DataFrame (df). It ensures consistency between the actual files and the dataset and identifies unused or duplicate files for potential removal.
# 1. Scanning Directories:
# The program starts by recursively scanning the root directory (root_dir) to collect paths of all NIFTI files (.nii, .nii.gz) and other files.
# The file paths are stored in two lists:
# nii_files: For NIFTI files.
# other_files: For non-NIFTI files.
# 2. Grouping NIFTI Files by ID:
# NIFTI files are grouped by extracting a unique ID from their file names using a regex pattern (e.g., CENTER_CLASSIDENTIFIER, like CZE_B01).
# Files with the same ID are grouped together. If a group contains:
# Two files: One with the keyword "MASKS" is identified as the annotation path, and the other is the image path. These are stored in a dictionary pairs with the ID as the key.
# More than two files: They are classified as duplicates and stored in the duplicates dictionary with the ID as the key.
# 3. Updating the DataFrame with Pairs:
# The program checks each entry in pairs:
# If both the image path and annotation path are already present in the same row of the dataset (df), the status of that row is set to 'Complete'. The entry is removed from pairs.
# If only the Tumor ID exists in the dataset but the paths don't match, the paths are updated in the dataset and the status is set to 'Auto-complete'. The entry is removed from pairs.
# If no matching Tumor ID exists in the dataset, the paths are printed as potentially removable files and are stored as remaining pairs.
# 4. Handling Duplicates:
# The program processes the entries in duplicates:
# For rows in the dataset without the status "Complete", if both the image and annotation paths match any of the files in a duplicates entry, the row's status is set to 'Complete'. These paths are removed from the duplicates entry.
# If no exact path match exists but a matching Tumor ID is found, the status is updated to 'Duplicates found! See duplicates_with_metadata dict for details!'. The entry is removed from duplicates and moved to the duplicates_with_metadata dictionary.
# Remaining unused duplicate paths are printed as potentially removable files and stored in the remaining_duplicates dictionary.
# 5. Handling Missing Data:
# For rows in the dataset that haven't been updated, the program sets their status to 'Missing'.
# 6. Printing unused files:
# After processing pairs and duplicates:
# The program prints Data without metadata (files not present in the dataset and thus potential candidates for removal).
# It also prints Unused duplicates (duplicate files not in use).
# This helps identify which files might be removed to clean up unused or redundant data.
# 7. Status Summary and File Saving:
# The program prints a count of all status values in the dataset (Complete, Auto-complete, Duplicates found, Missing, etc.).
# The updated dataset is saved as datset_updates.csv.

# Function to recursively scan the directory for files
def recursive_scan(root_dir):
    nii_files = []
    other_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            relative_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
            if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                nii_files.append(relative_path)
            else:
                other_files.append(relative_path)
    return nii_files, other_files


# Function to extract ID from the file path
def extract_id(file_path):
    pattern = r'([A-Z]{3})_([A-Z])(\d+)'  # Regex pattern to extract ID
    match = re.search(pattern, file_path)
    if match:
        return f"{match.group(1)}_{match.group(2)}{match.group(3)}"
    return None


# Function to group and filter NIFTI files
def group_nifti_files_by_id(nii_files):
    pairs = {}
    duplicates = {}

    id_dict = {}
    for file in nii_files:
        file_id = extract_id(file)
        if file_id:
            if file_id not in id_dict:
                id_dict[file_id] = []
            id_dict[file_id].append(file)

    for file_id, paths in id_dict.items():
        if len(paths) == 2:
            if any('MASKS' in path for path in paths):
                image_path = [path for path in paths if 'MASKS' not in path][0]
                annotation_path = [path for path in paths if 'MASKS' in path][0]
                pairs[file_id] = [image_path, annotation_path]
        elif len(paths) > 2:
            duplicates[file_id] = paths

    return pairs, duplicates


# Function to update dataframe based on matched pairs
def update_dataframe_with_pairs(df, pairs):
    remaining_pairs = {}

    for tumor_id, (image_path, annotation_path) in pairs.items():
        matching_row = df[(df['Image path'] == image_path) & (df['Annotation path'] == annotation_path)]
        if not matching_row.empty:
            df.loc[matching_row.index, 'status'] = 'Complete'
        else:
            row_with_tumor_id = df[df['Tumor ID'] == tumor_id]
            if not row_with_tumor_id.empty:
                df.loc[row_with_tumor_id.index, 'Image path'] = image_path
                df.loc[row_with_tumor_id.index, 'Annotation path'] = annotation_path
                df.loc[row_with_tumor_id.index, 'status'] = 'Auto-complete'
            else:
                remaining_pairs[tumor_id] = [image_path, annotation_path]

    if remaining_pairs:
        with open('Data_without_metadata.txt', 'w') as f:
            for tumor_id, paths in remaining_pairs.items():
                f.write(f"{paths[0]}\n")
                f.write(f"{paths[1]}\n")

    return df


# Function to handle duplicates
def process_duplicates(df, duplicates):
    duplicates_with_metadata = {}
    remaining_duplicates = {}

    for tumor_id, paths in list(duplicates.items()):
        for index, row in df[df['status'] != 'Complete'].iterrows():
            if row['Image path'] in paths and row['Annotation path'] in paths:
                df.loc[index, 'status'] = 'Complete'
                paths.remove(row['Image path'])
                paths.remove(row['Annotation path'])

        if not df[(df['Tumor ID'] == tumor_id) & (df['status'] != 'Complete')].empty:
            df.loc[(df['Tumor ID'] == tumor_id) & (df['status'] != 'Complete'), 'status'] = 'Duplicates found! See duplicates_with_metadata dict for details!'
            duplicates_with_metadata[tumor_id] = paths
            duplicates.pop(tumor_id)
        else:
            if paths:
                remaining_duplicates[tumor_id] = paths

    if remaining_duplicates:
        with open('Unused_duplicates.txt', 'w') as f:
            for tumor_id, paths in remaining_duplicates.items():
                for path in paths:
                    f.write(f"{path}\n")

    return df, duplicates_with_metadata


# Main function to perform the tasks
def main(root_dir, df):
    # Step 1: Get the nii and other file lists
    nii_files, other_files = recursive_scan(root_dir)

    # Step 2: Group NIFTI files by ID
    pairs, duplicates = group_nifti_files_by_id(nii_files)

    # Step 3: Update df based on pairs
    df = update_dataframe_with_pairs(df, pairs)

    # Step 4: Process duplicates
    df, duplicates_with_metadata = process_duplicates(df, duplicates)

    # Step 5: Set missing status for rows with no status set
    df.loc[df['status'].isnull(), 'status'] = 'Missing'

    # Step 6: Print a count of all different statuses
    print("\nStatus counts:")
    print(df['status'].value_counts())

    # Save the updated dataframe
    df.to_csv('Updated dataset.csv', index=False)


# Usage
root_dir = '/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4'  # Specify the root dataset directory
df = pd.read_csv(os.path.join(root_dir, "dataset.csv"))
df['status'] = None  # Add 'status' column
main(root_dir, df)