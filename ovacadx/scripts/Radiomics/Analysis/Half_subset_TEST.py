import os
import argparse
import statistics
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import statistics
from tqdm.contrib.concurrent import process_map
#Radiomics imports
import sys
sys.path.insert(1, '..')


from feature_selection import load_data
from utils import get_dataset

import pickle

# Function to save a dictionary to a file
def save_dict_to_file(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

# Function to load a dictionary from a file
def load_dict_from_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


#This file will be used to search all feature spaces for a 50% subgroup that has high AUC.
#This will be done on the VALIDATION set, and then validated on the test set.

def get_args_parser():
    parser = argparse.ArgumentParser(description='Ensemble of nested crossvalidation results')
    parser.add_argument('--output_csv', type=str, default='/path/to/ test csv file') #Source CSV FILE WITH LOGITS
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--num_inner_splits', type=int, default=5,
                        help='Number of inner splits for nested cross-validation')
    parser.add_argument('--num_outer_splits', type=int, default=5,
                        help='Number of outer splits for nested cross-validation')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
    parser.add_argument('--plot', type=int, default=1,
                        help='Whether to also SHOW the AUC curves while running')
    # parser.add_argument("--model_name", type=str, default='UNDEFINED MODEL', help ="set label name for model")

    return parser


def main(args):
    col = "red"
    np.random.seed(args.seed)
    output_df_metric_statistics = None
    # Load results
    csv_path = os.path.join(args.data_dir,"preprocessed_data.csv")
    seed=0
    remove_outliers=False
    preprocessed_data = pd.read_csv(csv_path)

    mod = get_dataset(0,csv_path, seed, remove_outliers)
    radiomics,radiomics_test = load_data(args.data_dir,"",mod)
    radiomics = pd.concat([radiomics, radiomics_test], ignore_index=True) #Get radiomics of ALL samples regardless of whether they are in the test set. We do this because the splitting on test and val set is already in our results.csv, hence here we simply load all radiomic features and then filter using the IDs of results.csv.

    # Check the resulting dataframe
    print(radiomics.head())
    results_name = os.path.split(args.output_csv)[-1].split(".")[0]
    classifier = results_name.replace("results",'')
    if classifier == "NN":
        classifier = "MLP"
    results_test_path = args.output_csv
    args.data_path = os.path.split(args.output_csv)[0]
    results_test = pd.read_csv(results_test_path)

    results_test_ext = results_test[results_test['tumor_id'].str.startswith("AVL_")]
    results_test_int = results_test[~results_test['tumor_id'].str.startswith("AVL_")]
    results_val = pd.read_csv(os.path.join(args.data_path,"VAL_" + os.path.split(results_test_path)[-1]))

    AUC_curves={}

    results_test = results_test_int
    half_group_scores = {}
    half_group_side = {}
    current_feature_name = "original_shape_VoxelVolume"
    # Average the test set results of outer loop of nested cross validation.
    for ind, row in results_test.iterrows():
        lbl = int(preprocessed_data[preprocessed_data['tumor_id'] == row['tumor_id']]['label'].iloc[0])
        results_test.at[ind, 'label'] = lbl
        logits = []
        clean_row = row.drop('tumor_id')
        for i, (_, value) in enumerate(clean_row.items()):
            if not np.isnan(value):
                logits.append(value)
        results_test.at[ind, 'avg_pred'] = statistics.mean(logits)

    # current_feature_name=radiomics.columns[2:][0] #2 is here to skip ID and label
    pp_data_int = preprocessed_data[~preprocessed_data['tumor_id'].str.startswith("AVL_")].copy()
    pp_data_int = pp_data_int[(pp_data_int['label'] == 0) | (pp_data_int['label'] == 1)]
    pp_data_int.loc[:, 'current_feature'] = None
    for index, nid in pp_data_int['tumor_id'].items():
        matching_row = radiomics[radiomics['ID'] == nid]
        if not matching_row.empty:  # Ensure there's a match
            pp_data_int.at[index, 'current_feature'] = matching_row.iloc[0][current_feature_name]
    median_volume = np.median(pp_data_int['current_feature'])
    # median_volume = np.percentile(pp_data_int['current_feature'],75)
    small_group = pp_data_int[pp_data_int['current_feature'] < median_volume]
    large_group = pp_data_int[pp_data_int['current_feature'] >= median_volume]
    auc = {}
    for group, name in zip([pp_data_int, small_group, large_group], ["all", "small", "large"]):
        num_mal = len(group[group['label'] == 1])
        num_ben = len(group[group['label'] == 0])
        if num_ben < 10 or num_mal < 10 or num_ben / num_mal > 10 or num_mal / num_ben > 10:  # Abritrary limts on when AUC is reliable
            print("Cannot compute AUC reliably due to sample imbalance!")
        else:
            lbls = []
            preds = []
            for nid in group['tumor_id']:
                row_results_test = results_test[results_test['tumor_id'] == nid]
                preds.append(row_results_test['avg_pred'])
                lbls.append(row_results_test['label'])

        auc[name] = roc_auc_score(lbls, preds)
    auc_df = pd.DataFrame(list(auc.items()), columns=["Group", "AUC score"])
    # Set the value of radiomics[current_feature_name] where the value was "Max_50_AUC" to 0.8
    half_group_scores[current_feature_name] = max(
        auc_df['AUC score'][1:])  # We take 1: as we want to skip the score for "ALL"
    half_group_side[current_feature_name] = np.argmax(
        auc_df['AUC score'][1:])  # 0 for below threshold, 1 for above threshold
    print(half_group_scores)



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)