
import sys
sys.path.insert(1, '..')
from utils import count_center_class_combinations,count_center_patient_combinations, df_to_latex_table
import pandas as pd
import os

# data_dir = "/home/eloy/Documents/Data/LUNG_DATA/NIFTI-Processed_new"
data_dir = "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new"
preprocessed_data = pd.read_csv(os.path.join(data_dir,"preprocessed_data.csv"))


df = count_center_class_combinations(preprocessed_data)
print("NUMBER OF LESIONS BELOW")
print(df)
print(df_to_latex_table(df))
#


print("NUMBER OF PATIENT BELOW")
df = count_center_patient_combinations(preprocessed_data)
print(df)




#Count the number of patients:
print("as")
# data = {'tumor_id': ['CZE_B001', 'CZE_B002', 'US_A003', 'CZE_M001', 'US_A004']}
# df = pd.DataFrame(data)
#
# count_df = count_center_class_combinations(df)
# print(count_df)

import matplotlib.pyplot as plt
import numpy as np

# Data for easiest subset
easiest_subset_benign = [45, 6]  # BRE, CZE
easiest_subset_malignant = [8, 15]  # BRE, CZE
easiest_subset_benign_outliers = [35, 16]  # Outliers, Non-Outliers
easiest_subset_malignant_outliers = [0, 23]  # Outliers, Non-Outliers

# Data for full dataset
full_dataset_benign = [194, 61]  # BRE, CZE
full_dataset_malignant = [54, 61]  # BRE, CZE
full_dataset_benign_outliers = [90, 165]  # Outliers, Non-Outliers
full_dataset_malignant_outliers = [12, 103]  # Outliers, Non-Outliers

# Calculate percentages for stacked bars
def calculate_percentages(values):
    total = sum(values)
    return [(x / total) * 100 for x in values]

easiest_subset_benign_perc = calculate_percentages(easiest_subset_benign)
easiest_subset_malignant_perc = calculate_percentages(easiest_subset_malignant)
easiest_subset_benign_outliers_perc = calculate_percentages(easiest_subset_benign_outliers)
easiest_subset_malignant_outliers_perc = calculate_percentages(easiest_subset_malignant_outliers)

full_dataset_benign_perc = calculate_percentages(full_dataset_benign)
full_dataset_malignant_perc = calculate_percentages(full_dataset_malignant)
full_dataset_benign_outliers_perc = calculate_percentages(full_dataset_benign_outliers)
full_dataset_malignant_outliers_perc = calculate_percentages(full_dataset_malignant_outliers)

# Define bar positions
# bar_width = 0.35
# indices = np.arange(2)  # Benign and Malignant
#
# # Plotting
# fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
#
# # Plot for BRE and CZE
# ax[0].bar(indices - bar_width/2, easiest_subset_benign_perc, width=bar_width, label='Benign (Easiest Subset)')
# ax[0].bar(indices - bar_width/2, easiest_subset_malignant_perc, bottom=easiest_subset_benign_perc, width=bar_width, label='Malignant (Easiest Subset)')
# ax[0].bar(indices + bar_width/2, full_dataset_benign_perc, width=bar_width, label='Benign (Full Dataset)')
# ax[0].bar(indices + bar_width/2, full_dataset_malignant_perc, bottom=full_dataset_benign_perc, width=bar_width, label='Malignant (Full Dataset)')
#
# # Plot for Outliers and Non-Outliers
# ax[1].bar(indices - bar_width/2, easiest_subset_benign_outliers_perc, width=bar_width, label='Benign (Easiest Subset)')
# ax[1].bar(indices - bar_width/2, easiest_subset_malignant_outliers_perc, bottom=easiest_subset_benign_outliers_perc, width=bar_width, label='Malignant (Easiest Subset)')
# ax[1].bar(indices + bar_width/2, full_dataset_benign_outliers_perc, width=bar_width, label='Benign (Full Dataset)')
# ax[1].bar(indices + bar_width/2, full_dataset_malignant_outliers_perc, bottom=full_dataset_benign_outliers_perc, width=bar_width, label='Malignant (Full Dataset)')
#
# # Labels and titles
# ax[0].set_title('Source Distribution (BRE vs. CZE)')
# ax[0].set_xticks(indices)
# ax[0].set_xticklabels(['Benign', 'Malignant'])
# ax[0].set_ylabel('Percentage')
# ax[0].legend()
#
# ax[1].set_title('Outlier Status Distribution')
# ax[1].set_xticks(indices)
# ax[1].set_xticklabels(['Benign', 'Malignant'])
#
# plt.tight_layout()
# plt.show()


