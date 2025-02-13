import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os

import pandas as pd


def dataframe_to_vertical_latex(df, caption=None, label=None, index=True, column_format=None):
    """
    Convert a DataFrame into a LaTeX table with vertically stacked header text.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert.
    caption (str, optional): The caption for the table.
    label (str, optional): The label for referencing the table.
    index (bool, optional): Whether to include the DataFrame's index in the table. Default is True.
    column_format (str, optional): LaTeX column alignment (e.g., 'lcr' for three columns). If None, pandas' default is used.

    Returns:
    str: The LaTeX table as a string.
    """

    # Helper function to stack text vertically
    def stack_text(text):
        return r'\begin{tabular}[c]{@{}c@{}}' + r' \\ '.join(text.split()) + r'\end{tabular}'

    # Apply the stack_text function to all column headers
    df.columns = [stack_text(col) for col in df.columns]

    # Convert DataFrame to LaTeX
    latex_str = df.to_latex(index=index, column_format=column_format, escape=False)

    # Insert caption if provided
    if caption:
        latex_str = latex_str.replace('\\toprule', '\\toprule\n\\caption{' + caption + '}')

    # Insert label if provided
    if label:
        latex_str = latex_str.replace('\\bottomrule', '\\bottomrule\n\\label{' + label + '}')

    return latex_str

def get_args_parser():
    parser = argparse.ArgumentParser(description='Bootstrapping of nested crossvalidation results')
    parser.add_argument('--data_path_with_outliers', type=str, default='/path/to/results/csv')
    parser.add_argument('--data_path_without_outliers', type=str, default='/path/to/results/csv')
    parser.add_argument('--preprocessed_path', type=str, default = '/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new/')
    parser.add_argument('--num_bootstraps', type=int, default=100, help='Number of bootstraps')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')

    return parser


def main(args):
    #csv_files = [f for f in os.listdir(args.data_dir) if f.endswith('.csv')]
    # Load results
    # results = pd.read_csv(args.data_path)
    resulting_dfs=[]
    csv_pp = csv_path = os.path.join(args.preprocessed_path, "preprocessed_data.csv")
    full_set = pd.read_csv(csv_pp)
    full_set = full_set[full_set['label']!=2] #remove borderline
    outlier_set = full_set[full_set['outlier']==1]
    no_outlier_set = full_set[full_set['outlier'] == 0]
    #csv_name = "NN"
    list_of_resulting_dfs = []
    for classifier in ["NN","SVM","RF","LR"]:

        df_results = pd.DataFrame(columns=['Trained including outliers','Trained without outliers'])
        #classifier = csv_file.replace("results", "").replace(".csv", "")
        for o_set,set_name,idx in zip([full_set,outlier_set,no_outlier_set], ['Test including outliers','Test only outliers', 'Test without outliers'],[0,1,2]):
            # Set seed
            #
            print("Including outliers for:", classifier)
            if idx < 2:
                csv_file = os.path.join(args.data_path_with_outliers,"results_CS_" + "WITH_OUTLIERS_"+classifier+".csv")
            else:
                csv_file = os.path.join(args.data_path_with_outliers,
                                        "results_CS_" + classifier + ".csv")
            np.random.seed(args.seed)
            results = pd.read_csv(csv_file)
            results = results[results['tumor_id'].isin(o_set['tumor_id'])]
            aucs_internal = []
            aucs_external = []
            for _ in range(args.num_bootstraps):
                labels_internal = []
                logits_internal = []
                labels_external = []
                logits_external = []
                for _, row in results.iterrows():
                    label = 1 if 'M' in row['tumor_id'].split('_')[1] else 0
                    labels_external.append(label) if 'AVL' in row['tumor_id'] else labels_internal.append(label)
                    # random pick one of the columns that is not 'tumor_id' and does not contain nan
                    logit = row[1:].dropna().sample().item()
                    logits_external.append(logit) if 'AVL' in row['tumor_id'] else logits_internal.append(logit)
                aucs_internal.append(roc_auc_score(labels_internal, logits_internal))
                aucs_external.append(roc_auc_score(labels_external, logits_external))
        #Summarize AUCs here
            df_results.at[set_name, "Trained including outliers"] = list(aucs_internal)
        for o_set,set_name,idx in zip([full_set,outlier_set,no_outlier_set], ['Test including outliers','Test only outliers', 'Test without outliers'],[0,1,2]):
            # Set seed
            #
            print("Excluding outliers for:", classifier)
            if idx < 2:
                csv_file = os.path.join(args.data_path_without_outliers,"results_CS_" + "WITH_OUTLIERS_"+classifier+".csv")
            else:
                csv_file = os.path.join(args.data_path_without_outliers,
                                        "results_CS_" + classifier + ".csv")
            np.random.seed(args.seed)
            results = pd.read_csv(csv_file)
            results = results[results['tumor_id'].isin(o_set['tumor_id'])]
            aucs_internal = []
            aucs_external = []
            for _ in range(args.num_bootstraps):
                labels_internal = []
                logits_internal = []
                labels_external = []
                logits_external = []
                for _, row in results.iterrows():
                    label = 1 if 'M' in row['tumor_id'].split('_')[1] else 0
                    labels_external.append(label) if 'AVL' in row['tumor_id'] else labels_internal.append(label)
                    # random pick one of the columns that is not 'tumor_id' and does not contain nan
                    logit = row[1:].dropna().sample().item()
                    logits_external.append(logit) if 'AVL' in row['tumor_id'] else logits_internal.append(logit)
                aucs_internal.append(roc_auc_score(labels_internal, logits_internal))
                aucs_external.append(roc_auc_score(labels_external, logits_external))
        #Summarize AUCs here
            df_results.at[set_name, "Trained without outliers"] = list(aucs_internal)

        list_of_resulting_dfs.append(df_results)

    # import pandas as pd
    # import numpy as np

    # Assuming list_of_resulting_dfs is a list of DataFrames with the same structure
    # // Example list_of_resulting_dfs = [df1, df2, df3, ...]

    # Step 1: Initialize an empty DataFrame to hold the results
    aggregated_df = pd.DataFrame()

    # Step 2: Combine all the values from the corresponding cells of the DataFrames
    for col in list_of_resulting_dfs[0].columns:
        for idx in list_of_resulting_dfs[0].index:
            all_values = []
            for df in list_of_resulting_dfs:
                all_values.extend(df.loc[idx, col])

            # Calculate median, 25th percentile, and 75th percentile
            median = np.median(all_values)
            p25 = np.percentile(all_values, 25)
            p75 = np.percentile(all_values, 75)

            # Store the result as a formatted string
            aggregated_df.loc[idx, col] = f"{median:.3f}, ({p25:.3f}, {p75:.3f})"

    # Display the resulting aggregated dataframe
    print(aggregated_df)
    print(dataframe_to_vertical_latex(average_results))
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
