import sys
sys.path.insert(1, '..')
from Analysis.ens_no_logits2_sizes import main, get_args_parser
import pandas as pd

OVARY_DATA_DIR = "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new"
LUNG_DATA_DIR = "/home/eloy/Documents/Data/LUNG_DATA/NIFTI-Processed_new"
ovarian_test_results = "/home/eloy/Documents/Graduation_project/Code/MIL/experiments/experiment_ov_rotate/resultsMILCNN.csv"
lung_test_results="/home/eloy/Documents/Graduation_project/Code/MIL/experiments/experiment_lung_rotate/resultsMILCNN.csv"




parser = get_args_parser()
args = parser.parse_args()


args.data_dir = OVARY_DATA_DIR
args.output_csv = ovarian_test_results
# print(args)
print("Ovarian results BELOW")
result_1=main(args)

# print(result_1)

args.data_dir = LUNG_DATA_DIR
args.output_csv = lung_test_results
print("Lung results BELOW")
result_2=main(args)

# # print(result_2)
#
result_1_df = pd.DataFrame(result_1)
result_2_df = pd.DataFrame(result_2)
#


# Create DataFrames with "Size" column
result_1_df = pd.DataFrame(result_1, index=['small', 'large', 'all'])
result_1_df = result_1_df.reset_index().rename(columns={'index': 'Size'})
result_1_df['Dataset'] = 'Ovarian'

result_2_df = pd.DataFrame(result_2, index=['small', 'large', 'all'])
result_2_df = result_2_df.reset_index().rename(columns={'index': 'Size'})
result_2_df['Dataset'] = 'LIDC'

# Combine the two DataFrames
merged_df = pd.concat([result_1_df, result_2_df], axis=0)

# Reorder the columns to have 'Dataset' as the first column
merged_df = merged_df[['Dataset', 'Size', 'AUC', 'Total ACC', 'Benign ACC', 'Malignant ACC', '#B', '#M']]

# Reset the index to make it a regular column
merged_df = merged_df.reset_index(drop=True)
merged_df['Size'] = merged_df['Size'].replace({'small': r'$<\text{Median}$', 'large': r'$\geq\text{Median}$', 'all':'All'})



#Rounding and precision

# Convert '#B' and '#M' to int
merged_df['#B'] = merged_df['#B'].astype(int)
merged_df['#M'] = merged_df['#M'].astype(int)

# Limit the precision of other float columns to 3 digits
float_columns = ['AUC', 'Total ACC', 'Benign ACC', 'Malignant ACC']
merged_df[float_columns] = merged_df[float_columns].round(3)

# Display the result
print(merged_df)

# # Merge the two DataFrames on the 'Group' column
# merged_df = pd.merge(result_1_df, result_2_df, on='Group')
#
# # Display the final merged DataFrame
# print(merged_df)
#
# latex_table = df.to_latex(float_format="%.4f", caption="Performance Metrics Table", label="tab:metrics")
#
# # Print the LaTeX table
# print(latex_table)

import re

# Generate the LaTeX table
latex_table = merged_df.rename(columns=lambda x: x.replace('#', '\\#')).to_latex(
    float_format="%.3f",
    caption="Performance Metrics for Ovarian and LIDC Datasets",
    label="tab:performance_metrics",
    header=True,
    index=False,
    escape=False,
    multirow=False,
    column_format='c c c c c c c c'  # Adding vertical dividers
)

# Match the last "Ovarian" row and insert \hline after it
latex_table = re.sub(r'(\nOvarian[^\n]*\n)(?=\s*LIDC)', r'\1\\hline\n', latex_table)

# Print the modified LaTeX table
print(latex_table)
#What if we repeat this experiment for ALL radiomic features.
#Then take a look at the ones with the highest performance subgroup
#Then perform a range test on the threshold (instead of median).
#Then plot a graph of performance vs feature


