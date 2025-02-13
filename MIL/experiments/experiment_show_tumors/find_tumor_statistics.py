import os
import pandas as pd
import SimpleITK as sitk
import numpy as np

# Define the data directory and load the dataset
data_dir = '/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new'
csv_path = os.path.join(data_dir, 'preprocessed_data.csv')
dataset = pd.read_csv(csv_path)

# Function to find the slice with maximal annotation
def get_max_annotation_area(annotation):
    max_area = 0
    for slice_index in range(annotation.GetSize()[2]):  # Iterate through slices
        annotation_slice = sitk.GetArrayFromImage(annotation[:, :, slice_index])
        area = np.sum(annotation_slice)
        if area > max_area:
            max_area = area
    return max_area

# Function to process the dataset and return information
def process_dataset(data_dir, dataset):
    # Initialize list to store results
    results_list = []

    for index, row in dataset.iterrows():
        tumor_id = row['tumor_id']
        label = row['label']
        image_path = os.path.join(data_dir, row['image_path'])
        annotation_path = os.path.join(data_dir, row['annot_path'])

        # Read image and annotation
        image = sitk.ReadImage(image_path)
        annotation = sitk.ReadImage(annotation_path)

        # Get voxel spacing
        voxel_spacing = image.GetSpacing()

        # Find maximum annotation area
        max_area = get_max_annotation_area(annotation)

        # Append information to the results list
        results_list.append({'Tumor ID': tumor_id,
                             'Label': label,
                             'Voxel Spacing': voxel_spacing,
                             'Max Area': max_area})

    # Create DataFrame from the results list
    results = pd.DataFrame(results_list)

    # Calculate minimum and maximum voxel spacing
    voxel_spacings = np.array(results['Voxel Spacing'].tolist())
    min_voxel_spacing = voxel_spacings.min(axis=0)
    max_voxel_spacing = voxel_spacings.max(axis=0)

    # Sort dataframe by maximum area
    results = results.sort_values(by='Max Area', ascending=False)


    results.to_csv("Results.csv")
    # Find largest and smallest tumors for both labels 'B' and 'M'
    largest_tumor_B = results[results['Label'] == 0].iloc[0]
    smallest_tumor_B = results[results['Label'] == 0].iloc[-1]
    largest_tumor_M = results[results['Label'] == 1].iloc[0]
    smallest_tumor_M = results[results['Label'] == 1].iloc[-1]

    mean_voxel_spacing = np.mean(voxel_spacings, axis=0)

    # Calculate interquartile range (IQR)
    iqr_voxel_spacing = np.percentile(voxel_spacings, 75, axis=0) - np.percentile(voxel_spacings, 25, axis=0)

    # Print the results
    print(f"Mean: {mean_voxel_spacing}")
    print(f"Interquartile Range (IQR): {iqr_voxel_spacing}")

    return min_voxel_spacing, max_voxel_spacing, largest_tumor_B, smallest_tumor_B, largest_tumor_M, smallest_tumor_M

# Call the function to process the dataset
min_voxel_spacing, max_voxel_spacing, largest_tumor_B, smallest_tumor_B, largest_tumor_M, smallest_tumor_M = process_dataset(data_dir, dataset)

# Display the results
print("Minimum Voxel Spacing:", min_voxel_spacing)
print("Maximum Voxel Spacing:", max_voxel_spacing)
print("Largest Tumor (Label B):", largest_tumor_B)
print("Smallest Tumor (Label B):", smallest_tumor_B)
print("Largest Tumor (Label M):", largest_tumor_M)
print("Smallest Tumor (Label M):", smallest_tumor_M)
