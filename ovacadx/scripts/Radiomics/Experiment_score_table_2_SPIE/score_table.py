import pandas as pd

import pandas as pd
import os





def collect_metrics_to_csv_and_latex_NEW(csv_paths, model_names, data_labels, output_csv_path, D):
    if not (len(csv_paths) == len(model_names) == len(data_labels)):
        raise ValueError("The number of CSV paths, model names, and data labels must match.")

    # Define the columns we are interested in
    columns_of_interest = [
        'AUC', 'Sensitivity', 'Specificity'
    ]

    # Initialize an empty list to store the results
    results_list = []

    # Iterate over the CSV files, model names, and data labels
    for csv_path, model_name, data_label in zip(csv_paths, model_names, data_labels):

        # Find ext file path
        path_parts = csv_path.split('/')
        filename = path_parts[-1]
        new_filename = 'ext' + filename
        ext_path = '/'.join(path_parts[:-1] + [new_filename])
        ext = os.path.exists(ext_path)
        if ext:
            df_ext = pd.read_csv(ext_path)
            df_ext.columns.values[0] = 'Metric'

        # Read the current CSV file
        df = pd.read_csv(csv_path)
        df.columns.values[0] = 'Metric'
        # Initialize a dictionary to store the results for the current model
        model_results = {'Data': data_label, 'Model': model_name}

        # Iterate over the metrics of interest and extract the required statistics
        for metric in columns_of_interest:
            median = f"{df.loc[df['Metric'] == metric, 'Median'].values[0]:.{D}f}"
            percentile_25th = f"{df.loc[df['Metric'] == metric, '25th Percentile'].values[0]:.{D}f}"
            percentile_75th = f"{df.loc[df['Metric'] == metric, '75th Percentile'].values[0]:.{D}f}"
            model_results[
                f"{metric} (IQR)"] = f"{median} (\\textcolor{{gray}}{{{percentile_25th}}} - \\textcolor{{gray}}{{{percentile_75th}}})"

        if ext:
            auc_ext_median = f"{df_ext.loc[df_ext['Metric'] == 'AUC', 'Median'].values[0]:.{D}f}"
            model_results['ext. AUC'] = auc_ext_median
        else:
            model_results['ext. AUC'] = "N/A"

        # Append the results for the current model to the results list
        results_list.append(model_results)

    # Convert the results list to a DataFrame
    results = pd.DataFrame(results_list)

    # Save the results to a CSV file
    results.to_csv(output_csv_path, index=False)

    # Convert the results DataFrame to LaTeX format
    latex_table = "\\begin{tabular}{llllll}\n\\toprule\n" \
                  "Data & Model & AUC (IQR) & Sensitivity (IQR) & Specificity (IQR) & ext. AUC \\\\\n\\midrule\n"

    current_data = None
    for i, row in results.iterrows():
        data = row['Data']
        if data != current_data:
            latex_table += f"{data}"
            current_data = data
        else:
            latex_table += "  "
        latex_table += f" & {row['Model']} & {row['AUC (IQR)']} & {row['Sensitivity (IQR)']} & {row['Specificity (IQR)']} & {row['ext. AUC']} \\\\\n"
        if i < len(results) - 1 and results.iloc[i + 1]['Data'] != data:
            latex_table += "\\midrule\n"

    latex_table += "\\bottomrule\n\\end{tabular}"

    # print(latex_table)
    return latex_table
def collect_metrics_to_csv_and_latex(csv_paths, model_names, data_labels, output_csv_path, D):
    raise ValueError("This is the old function. only here for archival purposes")
    # Check if the lengths of csv_paths, model_names, and data_labels match
    if not (len(csv_paths) == len(model_names) == len(data_labels)):
        raise ValueError("The number of CSV paths, model names, and data labels must match.")

    # Define the columns we are interested in
    columns_of_interest = [
        'AUC', 'Sensitivity', 'Specificity'
    ]

    # Initialize an empty list to store the results
    results_list = []

    # Iterate over the CSV files, model names, and data labels
    for csv_path, model_name, data_label in zip(csv_paths, model_names, data_labels):

        #Find ext file path
        # Split the input path to isolate the filename
        path_parts = csv_path.split('/')
        filename = path_parts[-1]
        # Insert 'ext' before the filename
        new_filename = 'ext' + filename
        # Reconstruct the new path
        ext_path = '/'.join(path_parts[:-1] + [new_filename])
        if os.path.exists(ext_path):
            ext=True
        else:
            ext=False
        if ext:
            df_ext = pd.read_csv(ext_path)
            df_ext.columns.values[0] = 'Metric'

        # Read the current CSV file
        df = pd.read_csv(csv_path)
        df.columns.values[0] = 'Metric'
        # Initialize a dictionary to store the results for the current model
        model_results = {'Model name': model_name, 'Data': data_label}

        # Iterate over the metrics of interest and extract the required statistics
        for metric in columns_of_interest:
            median = f"{df.loc[df['Metric'] == metric, 'Median'].values[0]:.{D}f}"
            percentile_25th = f"{df.loc[df['Metric'] == metric, '25th Percentile'].values[0]:.{D}f}"
            percentile_75th = f"{df.loc[df['Metric'] == metric, '75th Percentile'].values[0]:.{D}f}"
            model_results[f"{metric} (IQR)"] = f"{median} (\\textcolor{{gray}}{{{percentile_25th}}} - \\textcolor{{gray}}{{{percentile_75th}}})"
        if ext:
            auc_ext_median = f"{df_ext.loc[df_ext['Metric'] == 'AUC', 'Median'].values[0]:.{D}f}"
            model_results['ext. AUC'] = auc_ext_median
        else:
            model_results['ext. AUC'] = "N/A"

        # Append the results for the current model to the results list
        results_list.append(model_results)

    # Convert the results list to a DataFrame
    results = pd.DataFrame(results_list)

    # Save the results to a CSV file
    results.to_csv(output_csv_path, index=False)

    # Convert the results DataFrame to LaTeX format
    latex_table = results.to_latex(index=False, escape=False)
    #print(latex_table)
    return latex_table


# Example usage
csv_paths = [
'/home/eloy/Documents/Graduation_project/Code/ovacadx/scripts/Radiomics/Experiment_radiomics_SPIE/Radiomics_ovarian/resultsNN_compiled_summary.csv',
'/home/eloy/Documents/Graduation_project/Code/MIL/experiments/experiment_3DCNN_ovarian/resultsMILCNN_compiled_summary.csv',
'/home/eloy/Documents/Graduation_project/Code/MIL/experiments/experiment_3DCNN_OvarianRotate/resultsMILCNN_compiled_summary.csv',
'/home/eloy/Documents/Graduation_project/Code/MIL/experiments/experiment_ovarian_baseline/resultsMILCNN_compiled_summary.csv',
'/home/eloy/Documents/Graduation_project/Code/MIL/experiments/experiment_ov_rotate/resultsMILCNN_compiled_summary.csv',


'/home/eloy/Documents/Graduation_project/Code/ovacadx/scripts/Radiomics/Experiment_radiomics_SPIE/Radiomics_Lung/resultsNN_compiled_summary.csv',
'/home/eloy/Documents/Graduation_project/Code/MIL/experiments/experiment_3DCNN_LUNG/resultsMILCNN_compiled_summary.csv',
'/home/eloy/Documents/Graduation_project/Code/MIL/experiments/experiment_3DCNN_LIDCRotate/resultsMILCNN_compiled_summary.csv',
'/home/eloy/Documents/Graduation_project/Code/MIL/experiments/experiment_lung_baseline/resultsMILCNN_compiled_summary.csv',
'/home/eloy/Documents/Graduation_project/Code/MIL/experiments/experiment_lung_rotate/resultsMILCNN_compiled_summary.csv'



]
model_names = ['Radiomics+NN','3DCNN','3DCNN + R','MILCNN','MILCNN + R','Radiomics+NN','3DCNN','3DCNN + R','MILCNN','MILCNN + R']
data_labels = ['Ovarian','Ovarian','Ovarian','Ovarian','Ovarian','LIDC','LIDC','LIDC','LIDC','LIDC']
output_csv_path = 'combined_metrics.csv'

latex_table = collect_metrics_to_csv_and_latex_NEW(csv_paths, model_names, data_labels,output_csv_path,3)
print(latex_table)