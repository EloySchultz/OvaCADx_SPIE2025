import ovacadx.datasets as odata
import re
import pandas as pd
#
# def get_test_internal(csv_file,seed=42):
#     module = odata.OvarianTumorMILKFoldModule(
#         path=csv_file,
#         label="label",
#         orient=True,
#         mask_tumor=False,
#         k=0,  # which split or fold to load
#         num_splits=5,
#         batch_size=1,
#         num_workers=0,
#         pin_memory=False,
#         validation='internal',
#         # seed = seed,
#         key_internal_testset="is_internal_testset"
#     )
#     module.setup("test")
#     return module
#
#
# def get_test_external(csv_file,seed=42):
#     module = odata.OvarianTumorMILKFoldModule(
#         path=csv_file,
#         label="label",
#         orient=True,
#         mask_tumor=False,
#         k=0,  # which split or fold to load
#         num_splits=5,
#         batch_size=1,
#         num_workers=0,
#         pin_memory=False,
#         validation='external',
#         # seed=seed,
#         key_internal_testset="is_internal_testset"
#     )
#     module.setup("test")
#     return module
#
#
# def get_fold(k, csv_file,seed=42):
#     module = odata.OvarianTumorMILKFoldModule(
#         path=csv_file,
#         label="label",
#         orient=True,
#         mask_tumor=False,
#         k=k,  # which split or fold to load
#         num_splits=5,
#         batch_size=1,
#         num_workers=0,
#         pin_memory=False,
#         validation='internal',
#         # seed=seed,
#         key_internal_testset="is_internal_testset"
#     )
#     module.setup("fit")
#     return module


#if _randomized is used, you can get really high AUC by using seed=42 due to PURE LUCK
def get_dataset(k,csv_path,seed,remove_outliers):
    module = odata.MILNestedKFoldModule(
        path=csv_path,
        label="label",
        orient=True,
        mask_tumor=False,
        k=k,  # which split or fold to load
        num_inner_splits=5,
        num_outer_splits=5,
        batch_size=1,
        remove_outliers=remove_outliers,
        num_workers=0,
        pin_memory=False,
        seed = seed
    )
    module.setup("fit")
    return module


def filter_df(df, keep_ids):
    # Filter the DataFrame
    filtered_df = df[df['ID'].isin(keep_ids)]

    # Check if the length of the filtered DataFrame matches the length of keep_ids
    if len(filtered_df) == len(keep_ids):
        return filtered_df
    else:
        raise ValueError("The length of the filtered DataFrame does not match the length of keep_ids.")


def count_center_patient_combinations(df, remove_borderline=True): #Counts the number of PATIENTS, i.e. counts two tumors with similar ID as one.
    # Check for column name
    if 'tumor_id' in df.columns:
        col_name = 'tumor_id'
    elif 'ID' in df.columns:
        col_name = 'ID'
    else:
        raise ValueError("DataFrame must contain a 'tumor_id' or 'ID' column")

    # Extract Patient ID by removing side-specific identifiers (e.g., "_L" or "_R")
    def extract_patient_id(id):
        normalized_id = id.replace('-', '_')  # Replace '-' with '_'
        parts = normalized_id.split('_')
        if len(parts) == 3:  # Format like 'CZE_B001_L'
            return f"{parts[0]}_{parts[1]}"
        elif len(parts) == 2:  # Format like 'CZE_B001'
            return id
        return None

    # Create a new column for patient IDs
    df['Patient'] = df[col_name].apply(extract_patient_id)

    # Check for unmatched rows
    unmatched = df[df['Patient'].isnull()]
    if not unmatched.empty:
        print("Warning: Some IDs could not be matched to a patient pattern.")
        print("Unmatched rows:")
        print(unmatched)

    # Print which IDs are referred to the same patient
    patient_groups = df.groupby('Patient')[col_name].apply(list)
    for patient, ids in patient_groups.items():
        if len(ids) > 1:
            print(f"Patient ID '{patient}' refers to tumor IDs: {ids}")

    # Extract Center and Class using string operations
    df['Center'] = df[col_name].apply(lambda x: x.split('_')[0])
    df['Class'] = df[col_name].apply(lambda x: 'BL' if 'BL' in x else ('B' if 'B' in x else 'M'))

    # Drop borderline class if specified
    if remove_borderline:
        df = df[df['Class'] != 'BL']

    # Count unique patients grouped by center and class
    result = df.groupby(['Center', 'Class'])['Patient'].nunique().unstack(fill_value=0)

    # Add totals for rows and columns
    result.loc['Total'] = result.sum()
    result['Total'] = result.sum(axis=1)

    return result



def count_center_class_combinations(df, remove_borderline=True):
    # Check for column name
    if 'tumor_id' in df.columns:
        col_name = 'tumor_id'
    elif 'ID' in df.columns:
        col_name = 'ID'
    else:
        raise ValueError("DataFrame must contain a 'tumor_id' or 'ID' column")

    # If IDs start with "LIDC", count the classes based on the 'label' column
    if df[col_name][0].startswith("LIDC"):
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain a 'label' column for LIDC data.")

        # Map 0 to 'B' and 1 to 'M' in the label column
        df['label'] = df['label'].map({0: 'B', 1: 'M'})

        # Count occurrences of each class in the 'label' column
        result = df['label'].value_counts().to_frame(name='Count').T
        result['Total'] = result.sum(axis=1)

    else:
        # Extract Center and Class using regular expressions
        df['Center'] = df[col_name].apply(lambda x: re.match(r'^([A-Z]+)_', x).group(1))
        df['Class'] = df[col_name].apply(lambda x: re.match(r'^[A-Z]+_([A-Z]+)\d+', x).group(1))

        # Create a pivot table
        result = df.pivot_table(index='Center', columns='Class', aggfunc='size', fill_value=0)
        if remove_borderline:
            result = result.drop(['BL'], axis=1)

        # Replace 0 with 'B' and 1 with 'M'
        result = result.rename(columns={0: 'B', 1: 'M'})

        # Add a total row
        result.loc['Total'] = result.sum()

        # Add a total column
        result['Total'] = result.sum(axis=1)

    return result

def df_to_latex_table(df):
    # Convert the DataFrame to a LaTeX table string
    latex_table = df.to_latex(index=True, bold_rows=True)

    return latex_table