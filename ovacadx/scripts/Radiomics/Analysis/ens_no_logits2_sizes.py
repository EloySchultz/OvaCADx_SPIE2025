
#This file is used in experiment_compare_sizes, hence it is used for generating a table with volume as group seperator.
import os
import argparse
import statistics

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
import statistics
#Difference between this one and the original is the way the cofidence intervals are plotted on the ROC curves.
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



#TODO: fix that this script only does feature selection on the validation set instead.

def main(args):
    col = "red"
    np.random.seed(args.seed)
    output_df_metric_statistics = None
    # Load results
    preprocessed_data = pd.read_csv(os.path.join(args.data_dir,"preprocessed_data.csv"))
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

    for results_test, prefix in zip([results_test_int],[""]):
        print("Starting prefix" + prefix)
        if len(results_test)==0: #This happens for LIDC ext, as LIDC does not have an ext dataset.
            print(prefix+" does not hvae samples, skipping")
            continue;
        # print("Elloy", len(results_test))
        # group the test results per outer loop of the nested cross-validation
        mask = results_test.notnull()

        group_identifier = mask.apply(lambda row: tuple(row), axis=1)
        grouped = results_test.groupby(group_identifier)

        # get the thresholds for decision making per outer loop of the nested cross-validation
        # using the validation sets
        best_thresholds = []
        labels=[]

        best_thresholds=[]
        for _, row in results_val.iterrows():
            labels.append(int(preprocessed_data[preprocessed_data['tumor_id'] == row['tumor_id']]['label'].iloc[0]))
        # labels = [1 if 'M' in row['tumor_id'].split('_')[1] else 0 for _, row in results_val.iterrows()]

        if len(np.unique(labels)) < 2:
            print("Not enough unique labels, setting threshold to 0.5")
            best_thresholds = [0.5] * 25
        else:
            for column_name, column in results_val.items():
                if column_name == "tumor_id":
                    continue
                # get column values of all rows and belonging labels without nans
                column_list, labels_list = [], []
                for i, value in column.items():
                    if not np.isnan(value):
                        column_list.append(torch.tensor(value).item())
                        labels_list.append(labels[i])

                if len(column_list) == 0:
                    continue

                # calculate the roc curve and find the threshold that maximizes the f1 score
                fpr, tpr, thresholds = roc_curve(labels_list, column_list)
                f1 = 2 * tpr * (1 - fpr) / (tpr + 1 - fpr)
                best_threshold = thresholds[np.argmax(f1)]
                best_thresholds.append(best_threshold)


        #print("Thresholds:", best_thresholds) #Per fold.



        #Calculate AUC curve with uncertainty
        preds=[]
        labels=[]
        binary_pred=[]
        thresholds=[]
        for ind, row in results_test.iterrows():
            lbl = int(preprocessed_data[preprocessed_data['tumor_id'] == row['tumor_id']]['label'].iloc[0])
            labels.append(lbl)
            logits=[]
            clean_row = row.drop('tumor_id')
            for i, (_, value) in enumerate(clean_row.items()):
                if not np.isnan(value):
                    logits.append(value)
                    thresholds.append(best_thresholds[i])
            preds.append(statistics.mean(logits))
            results_test.at[ind, 'avg_pred'] = statistics.mean(logits)
            results_test.at[ind, 'label'] = lbl
            binary_predictions = [1 if p > t else 0 for p,t in zip(logits,thresholds)]
            binary_pred.append(np.mean(binary_predictions)>=0.5)

        preds = np.array(preds)
        labels = np.array(labels)
        binary_pred = np.array(binary_pred)
        bootstrap_aucs=[]
        bootstrap_fprs = []
        bootstrap_tprs = []
        bootstrap_binary = []

        accuracies=[]
        precisions = []
        recalls = []
        specificities=[]
        ppvs=[]
        npvs=[]
        sensitivities=[]
        # Interpolation base to ensure all FPRs are the same length
        interp_fpr = np.linspace(0, 1, 100)


        #For this experiment, we don't bootstrap. Instead we simply calculate the AUC over the three groups.



        pp_data_int = preprocessed_data[~preprocessed_data['tumor_id'].str.startswith("AVL_")]
        pp_data_int = pp_data_int[(pp_data_int['label']==0) | (pp_data_int['label']==1)]
        median_volume = np.median(pp_data_int['volume_cm3'])

        small_group = pp_data_int[pp_data_int['volume_cm3']<median_volume]
        large_group = pp_data_int[pp_data_int['volume_cm3'] >= median_volume]
        # benign_ids = pp_data_int[pp_data_int['label']==0]
        # benign_median_volume = np.median(benign_ids['volume_cm3'])
        # benign_ids_small = benign_ids[(benign_ids['volume_cm3'] < benign_median_volume)]
        # benign_ids_large = benign_ids[(benign_ids['volume_cm3'] >= benign_median_volume)]
        #
        # malignant_ids = pp_data_int[pp_data_int['label'] == 1]
        # malignant_median_volume = np.median(malignant_ids['volume_cm3'])
        #
        # malignant_ids_small = malignant_ids[malignant_ids['volume_cm3'] < malignant_median_volume]
        # malignant_ids_large = malignant_ids[malignant_ids['volume_cm3'] >= malignant_median_volume]
        auc={}

        for group,name in zip([small_group, large_group,pp_data_int],["small","large","all"]):
            num_mal=len(group[group['label']==1])
            num_ben = len(group[group['label'] == 0])
            if num_ben < 10 or num_mal<10 or num_ben/num_mal>10 or num_mal/num_ben>10: #Abritrary limts on when AUC is reliable
                print("Cannot compute AUC reliably due to sample imbalance!")
            else:
                lbls=[]
                preds=[]
                for nid in group['tumor_id']:
                    row_results_test = results_test[results_test['tumor_id'] == nid]
                    preds.append(row_results_test['avg_pred'])
                    lbls.append(row_results_test['label'])

            # Calculate thresholds based on validation set, which has been subdivided based on group (determined with median filtering)
            #We are going to calculate accuracy, which needs thresholds based on the validation set
            results_val_group = results_val[results_val['tumor_id'].isin(group['tumor_id'])].copy()
            results_val_group['avg_pred'] = results_val_group.drop(columns=['tumor_id']).mean(axis=1)

            # Read GT labels from preprocessed data directly (Woah df.map is really cool!)
            tumor_label_map = preprocessed_data.set_index('tumor_id')['label'].to_dict()
            # Map labels to df
            results_val_group['label'] = results_val_group['tumor_id'].map(tumor_label_map)
            fpr, tpr, thresholds = roc_curve(results_val_group['label'], results_val_group['avg_pred'])
            f1 = 2 * tpr * (1 - fpr) / (tpr + 1 - fpr)
            best_threshold = thresholds[np.argmax(f1)]
            results_val_group['avg_binary_pred'] = results_val_group['avg_pred'] > best_threshold
            # Compute confusion matrix using scipy (or sklearn)
            conf_matrix = confusion_matrix(results_val_group['label'], results_val_group['avg_binary_pred'], labels=[0, 1])
            total_accuracy = np.diag(conf_matrix).sum()/conf_matrix.sum()
            # Calculate accuracy per class
            accuracy_per_class = np.diag(conf_matrix) / conf_matrix.sum(axis=1)
            accuracy_per_class = np.nan_to_num(accuracy_per_class)  # Replace NaN with 0 (if any)
            # Create a DataFrame to display the results
            confusion_df = pd.DataFrame(conf_matrix, columns=['Pred benign', 'Pred malignant'],
                                        index=['True benign', 'True malignant'])

            # print("Confusion Matrix:")
            # print(confusion_df)
            # print("Total accuracty:",total_accuracy)
            # print("\nAccuracy Per Class:")
            # for label, accuracy in zip(confusion_df.index, accuracy_per_class):
            #     print(f"{label}: {accuracy:.2f}")

            #


            #fIN    ALLY, CALCULATE AUC DIRECTLY
            auc[name] = [roc_auc_score(lbls,preds),total_accuracy,accuracy_per_class[0],accuracy_per_class[1], num_ben,num_mal]



        auc_df = pd.DataFrame.from_dict(auc)
        auc_df=auc_df.transpose()
        auc_df.columns=['AUC','Total ACC','Benign ACC', 'Malignant ACC', '#B','#M']
        #print(auc_df)



    return auc_df


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)