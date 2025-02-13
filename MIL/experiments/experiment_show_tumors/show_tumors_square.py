import os
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours



font_size=55#26
# Define the data directory and load the dataset
data_dir = '/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new'
csv_path = os.path.join(data_dir, 'preprocessed_data.csv')
dataset = pd.read_csv(csv_path)

# IDs to plot (select the first four to fit in a 2x2 grid)
plot_ids = ['BRE_B209', 'BRE_B123', 'CZE_M34', 'BRE_M045']  # Adjusted to 4 tumor IDs

# Function to find the slice with maximal annotation
def get_max_annotation_slice(image, annotation):
    max_area = 0
    max_slice_index = 0
    for slice_index in range(annotation.GetSize()[2]):  # Iterate through slices
        annotation_slice = sitk.GetArrayFromImage(annotation[:, :, slice_index])
        area = np.sum(annotation_slice)
        if area > max_area:
            max_area = area
            max_slice_index = slice_index
    return max_slice_index

# Prepare plot for 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Flatten the axes for easier indexing
axes = axes.flatten()

# Process each ID
for idx, tumor_id in enumerate(plot_ids):
    row = dataset[dataset['tumor_id'] == tumor_id].iloc[0]
    image_path = os.path.join(data_dir, row['image_path'])
    annotation_path = os.path.join(data_dir, row['annot_path'])

    # Read image and annotation
    image = sitk.ReadImage(image_path)
    annotation = sitk.ReadImage(annotation_path)

    # Find slice with maximal annotation
    max_slice_index = get_max_annotation_slice(image, annotation)

    # Get the corresponding image and annotation slice
    image_slice = sitk.GetArrayFromImage(image[:, :, max_slice_index])
    annotation_slice = sitk.GetArrayFromImage(annotation[:, :, max_slice_index])

    # Clip the image between -200 and 300
    image_slice = np.clip(image_slice, -200, 300)

    # Create an outline of the annotation
    contours = find_contours(annotation_slice, 0.5)
    col = 'lime'
    title = "Benign"
    if idx > 1:  # Adjust to label the malignant tumors in the second row
        title = "Malignant"
        col = 'r'

    # Plot image slice
    axes[idx].imshow(image_slice, cmap='gray', vmin=-200, vmax=300)
    # Plot annotation contours
    for contour in contours:
        axes[idx].plot(contour[:, 1], contour[:, 0], col, linewidth=2)

    # Add bold letters (A, B, C, D) in Times New Roman at the top left
    axes[idx].text(12, 50+20*(font_size==55), chr(65 + idx), fontsize=font_size, fontweight='bold', color='white', fontname='Times New Roman')

    #axes[idx].set_title(f'{title}', fontsize=30, fontweight='bold', pad=20, fontname='Times New Roman')
    axes[idx].axis('off')

# Hide any unused subplots
for i in range(len(plot_ids), 4):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("data.pdf")
plt.show()