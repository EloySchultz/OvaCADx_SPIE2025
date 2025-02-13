import os
import argparse
import statistics

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
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

    for results_test, prefix in zip([results_test_int, results_test_ext,results_val],["","ext","val"]):
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
        for _, row in results_val.iterrows():
            labels.append(int(preprocessed_data[preprocessed_data['tumor_id']==row['tumor_id']]['label'].iloc[0]))
        # labels = [1 if 'M' in row['tumor_id'].split('_')[1] else 0 for _, row in results_val.iterrows()]

        if len(np.unique(labels)) <2:
            print("Not enough unique labels, setting threshold to 0.5")
            best_thresholds = [0.5]*25
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

        print("Thresholds:", best_thresholds) #Per fold.



        #Calculate AUC curve with uncertainty
        preds=[]
        labels=[]
        binary_pred=[]
        thresholds=[]
        for _, row in results_test.iterrows():
            labels.append(int(preprocessed_data[preprocessed_data['tumor_id'] == row['tumor_id']]['label'].iloc[0]))
            logits=[]
            clean_row = row.drop('tumor_id')
            for i, (_, value) in enumerate(clean_row.items()):
                if not np.isnan(value):
                    logits.append(value)
                    thresholds.append(best_thresholds[i])
            preds.append(statistics.mean(logits))
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
        # Bootstrapping
        for i in range(1000):
            # Generate a bootstrap sample
            indices = np.random.choice(len(preds), size=int(len(preds) / 5), replace=True)
            bootstrap_preds = preds[indices]
            bootstrap_labels = labels[indices]
            bootstrap_binary = binary_pred[indices]

            # Calculate the ROC curve for the bootstrap sample
            if len(np.unique(bootstrap_labels)) > 1:  # Ensure that there are both classes in the sample
                fpr, tpr, _ = roc_curve(bootstrap_labels, bootstrap_preds)
                auc = roc_auc_score(bootstrap_labels, bootstrap_preds)
                bootstrap_aucs.append(auc)

                # Interpolate TPR
                interp_tpr = np.interp(interp_fpr, fpr, tpr)
                interp_tpr[0] = 0.0  # Ensure the first value is 0.0
                bootstrap_fprs.append(interp_fpr)
                bootstrap_tprs.append(interp_tpr)

                #Calculate confusion matrix
                tn, fp, fn, tp = confusion_matrix(bootstrap_labels, bootstrap_binary).ravel()
                accuracies.append((tp + tn) /   (tp + tn + fp + fn))
                precisions.append(tp / (tp + fp) if (tp + fp) != 0 else 0)
                recalls.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
                specificities.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
                ppvs.append(tp / (tp + fp) if (tp + fp) != 0 else 0)  # Positive Predictive Value (PPV) is the same as precision
                npvs.append(tn / (tn + fn) if (tn + fn) != 0 else 0)
                sensitivities.append(tp / (tp + fn) if (tp + fn) != 0 else 0)  # Sensitivity is the same as recall
            else:
                # If there is only one class in the bootstrap sample, skip this iteration
                continue

        #Create a table for threshold-based metrics
        # Dictionary to store metric names and their values
        metrics = {
            'AUC': bootstrap_aucs,
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'Specificity': specificities,
            'Sensitivity': sensitivities,
            'PPV': ppvs,
            'NPV': npvs
        }

        # Function to calculate statistics
        def calculate_statistics(data):
            return {
                'Mean': np.mean(data),
                'Median': np.median(data),
                '25th Percentile': np.percentile(data, 25),
                '75th Percentile': np.percentile(data, 75),
                'Min': np.min(data),
                'Max': np.max(data)
            }

        # Calculate statistics for each metric
        metric_statistics = {metric: calculate_statistics(values) for metric, values in metrics.items()}

        # Create a DataFrame from the statistics
        df_metric_statistics = pd.DataFrame(metric_statistics).T

        if prefix!="ext":
            # Display the DataFrame
            print(prefix)
            print(df_metric_statistics)
            df_metric_statistics.to_csv(os.path.join(args.data_path,prefix+results_name+"_compiled_summary.csv"))
            print("SAVED TO ", os.path.join(args.data_path,prefix+results_name+"_compiled_summary.csv"))
            # Convert lists to arrays for easier manipulation
            bootstrap_fprs = np.array(bootstrap_fprs)
            bootstrap_tprs = np.array(bootstrap_tprs)

            # Calculate median and percentiles for TPR
            median_tpr = np.median(bootstrap_tprs, axis=0)
            tpr_25th_percentile = np.percentile(bootstrap_tprs, 25, axis=0)
            tpr_75th_percentile = np.percentile(bootstrap_tprs, 75, axis=0)
            med_auc = round(df_metric_statistics['Median']['AUC'],2)



            # Plotting
            plt.figure(figsize=(8, 6))
            plt.grid()
            plt.plot(interp_fpr, median_tpr, label='Median ROC (AUC = '+str(med_auc)+')', color='blue')
            plt.fill_between(interp_fpr, tpr_25th_percentile, tpr_75th_percentile, color='blue', alpha=0.2,
                             label='IQR (25th-75th)')

            plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Bootstrap ROC Curves')
            plt.legend(loc='best')
            plt.savefig(os.path.join(args.data_path, prefix+results_name + '_roc_curve_mean.png'), dpi=300,
                        bbox_inches='tight', transparent=True)
            #
            if args.plot==1:
                plt.show()
                #pass
        elif prefix=="ext":
            df_metric_statistics.to_csv(os.path.join(args.data_path, prefix + results_name + "_compiled_summary.csv"))
            print("SAVED TO ", os.path.join(args.data_path, prefix + results_name + "_compiled_summary.csv"))

        # Display median AUC and IQR
        median_auc = np.median(bootstrap_aucs)
        ci_lower = np.percentile(bootstrap_aucs, 25)
        ci_upper = np.percentile(bootstrap_aucs, 75)




        print(f"Median AUC: {median_auc}")
        print(f"IQR: [{ci_lower}, {ci_upper}]")
        if prefix == "":
            output_df_metric_statistics = df_metric_statistics
    return output_df_metric_statistics
    #
    # aucs = []
    # auc_ext = 0.
    # rocs = {'fpr': [], 'tpr': []}
    # roc_ext = {'fpr': [], 'tpr': []}
    # precisions = []
    # precisions_ext = 0.
    # recalls = []
    # recalls_ext = 0.
    # avg_precisions = []
    # avg_precisions_ext = 0.
    # # confusion matrix for every fold
    # cms = []
    # cm_ext = np.zeros((2, 2))
    # # auc_ext = 0.
    # for _, group in grouped:
    #
    #     labels = []
    #     probs = []
    #     preds = []
    #     for _, row in group.iterrows():
    #         labels.append(int(preprocessed_data[preprocessed_data['tumor_id'] == row['tumor_id']]['label'].iloc[0]))
    #
    #         # take all ther logit values and calculate the mean probability
    #         # ALSO: apply the threshold to the logit values
    #         logits = []
    #         threshlds = []
    #         clean_row = row.drop('tumor_id')
    #         for i, (_, value) in enumerate(clean_row.items()):
    #             if not np.isnan(value):
    #                 logits.append(value)
    #                 threshlds.append(best_thresholds[i])
    #         prob = [torch.tensor(logit).item() for logit in logits]
    #         probs.append(prob)
    #         preds.append([1 if prb > thres else 0 for prb, thres in zip(prob, threshlds)])
    #     probs = [statistics.mean(prob) for prob in probs]
    #     # preds = [1 if prob > 0.5 else 0 for prob in probs]
    #     # majority voting on predictions
    #     preds = [1 if sum(pred) > len(pred) / 2 else 0 for pred in preds]
    #     auc = roc_auc_score(labels, probs)
    #     fpr, tpr, _ = roc_curve(labels, probs)
    #     avg_precision = average_precision_score(labels, probs)
    #     precision = precision_score(labels, preds)
    #     recall = recall_score(labels, preds)
    #
    #     if not 'AVL' in row['tumor_id']:
    #         aucs.append(auc)
    #         rocs['fpr'].append(fpr)
    #         rocs['tpr'].append(tpr)
    #         precisions.append(precision)
    #         recalls.append(recall)
    #         avg_precisions.append(avg_precision)
    #         cms.append(np.array([[sum([1 for l, p in zip(labels, preds) if l == 0 and p == 0]),
    #                               sum([1 for l, p in zip(labels, preds) if l == 0 and p == 1])],
    #                              [sum([1 for l, p in zip(labels, preds) if l == 1 and p == 0]),
    #                               sum([1 for l, p in zip(labels, preds) if l == 1 and p == 1])]]))
    #     else:
    #         auc_ext = auc
    #         roc_ext['fpr'].append(fpr)
    #         roc_ext['tpr'].append(tpr)
    #         precisions_ext = precision
    #         recalls_ext = recall
    #         avg_precisions_ext = avg_precision
    #         cm_ext += np.array([[sum([1 for l, p in zip(labels, preds) if l == 0 and p == 0]),
    #                              sum([1 for l, p in zip(labels, preds) if l == 0 and p == 1])],
    #                             [sum([1 for l, p in zip(labels, preds) if l == 1 and p == 0]),
    #                              sum([1 for l, p in zip(labels, preds) if l == 1 and p == 1])]])
    # if len(roc_ext['fpr'])>0:
    #     ext=True
    # else:
    #     ext=False
    #
    # print("--------- Internal Test Set ---------")
    # print("AUC [MEDIAN][MEAN][IQR] [min - max]:", statistics.median(aucs),"[",statistics.mean(aucs),"]", "[", np.percentile(aucs, 25), "-", np.percentile(aucs, 75),
    #       "]", "[", min(aucs), "-", max(aucs), "]")
    # print("Precision [MEDIAN][MEAN][IQR] [min - max]:", statistics.median(precisions),"[",statistics.mean(precisions),"]", "[", np.percentile(precisions, 25), "-",
    #       np.percentile(precisions, 75), "]", "[", min(precisions), "-", max(precisions), "]")
    # print("Recall [MEDIAN][MEAN][IQR] [min - max]:", statistics.median(recalls),"[",statistics.mean(recalls),"]", "[", np.percentile(recalls, 25), "-",
    #       np.percentile(recalls, 75), "]", "[", min(recalls), "-", max(recalls), "]")
    # print("Average Precision [MEDIAN][MEAN][IQR] [min - max]:", statistics.median(avg_precisions),"[",statistics.mean(avg_precisions),"]","[",
    #       np.percentile(avg_precisions, 25), "-", np.percentile(avg_precisions, 75), "]", "[", min(avg_precisions), "-",
    #       max(avg_precisions), "]")
    # print("Specificity [MEDIAN][MEAN][IQR] [min - max]", statistics.median([cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in cms]), "[",statistics.mean([cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in cms]),"]", "[",
    #       np.percentile([cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in cms], 25), "-",
    #       np.percentile([cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in cms], 75), "]", "[",
    #       min([cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in cms]), "-",
    #       max([cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in cms]), "]")
    # print("Sensitivity [MEDIAN][MEAN][IQR] [min - max]:", statistics.median([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in cms]),"[",statistics.mean([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in cms]),"]", "[",
    #       np.percentile([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in cms], 25), "-",
    #       np.percentile([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in cms], 75), "]", "[",
    #       min([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in cms]), "-",
    #       max([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in cms]), "]")
    # print("PPV [MEDIAN][MEAN][IQR] [min - max]:", statistics.median([cm[1, 1] / (cm[1, 1] + cm[0, 1]) for cm in cms]),"[",statistics.mean([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in cms]), "]","[",
    #       np.percentile([cm[1, 1] / (cm[1, 1] + cm[0, 1]) for cm in cms], 25), "-",
    #       np.percentile([cm[1, 1] / (cm[1, 1] + cm[0, 1]) for cm in cms], 75), "]", "[",
    #       min([cm[1, 1] / (cm[1, 1] + cm[0, 1]) for cm in cms]), "-",
    #       max([cm[1, 1] / (cm[1, 1] + cm[0, 1]) for cm in cms]), "]")
    # print("NPV [MEDIAN][MEAN][IQR] [min - max]:", statistics.median([cm[0, 0] / (cm[0, 0] + cm[1, 0]) for cm in cms]),"[",statistics.mean([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in cms]),"]", "[",
    #       np.percentile([cm[0, 0] / (cm[0, 0] + cm[1, 0]) for cm in cms], 25), "-",
    #       np.percentile([cm[0, 0] / (cm[0, 0] + cm[1, 0]) for cm in cms], 75), "]", "[",
    #       min([cm[0, 0] / (cm[0, 0] + cm[1, 0]) for cm in cms]), "-",
    #       max([cm[0, 0] / (cm[0, 0] + cm[1, 0]) for cm in cms]), "]")
    # if ext:
    #     print("--------- External Test Set ---------")
    #     print("AUC:", auc_ext)
    #     print("Precision:", precisions_ext)
    #     print("Recall:", recalls_ext)
    #     print("Average Precision:", avg_precisions_ext)
    #     print("Specificity:", cm_ext[0, 0] / (cm_ext[0, 0] + cm_ext[0, 1]))
    #     print("Sensitivity:", cm_ext[1, 1] / (cm_ext[1, 0] + cm_ext[1, 1]))
    #     print("PPV:", cm_ext[1, 1] / (cm_ext[1, 1] + cm_ext[0, 1]))
    #     print("NPV:", cm_ext[0, 0] / (cm_ext[0, 0] + cm_ext[1, 0]))
    #
    # # Additional code to plot the ROC curve
    # all_labels = []
    # all_probs = []
    #
    # for _, group in grouped:
    #     labels = []
    #     probs = []
    #     for _, row in group.iterrows():
    #         labels.append(int(preprocessed_data[preprocessed_data['tumor_id'] == row['tumor_id']]['label'].iloc[0]))
    #
    #         logits = []
    #         clean_row = row.drop('tumor_id')
    #         for i, (_, value) in enumerate(clean_row.items()):
    #             if not np.isnan(value):
    #                 logits.append(value)
    #
    #         prob = [torch.tensor(logit).item() for logit in logits]
    #         probs.append(statistics.mean(prob))
    #
    #     all_labels.extend(labels)
    #     all_probs.extend(probs)
    #
    # # Calculate ROC curve
    # fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    # roc_auc = roc_auc_score(all_labels, all_probs)
    #
    # # Plot ROC curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='blue', lw=2, label=f'{classifier} (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label = 'Chance level (AUC = 0.5)',alpha=.8)
    # plt.title('ROC curve')
    # plt.xlabel('1 - Specificity (FPR)')
    # plt.ylabel('Sensitivity (TPR)')
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.legend(loc="lower right")
    # plt.grid(alpha=0.3)
    # plt.savefig(os.path.join(args.data_path, results_name + '_roc_curve_mean.png'), dpi=300, bbox_inches='tight', transparent=True)
    # # plt.show()
    #
    # # get tpr and fpr values for the average and std of roc curves with different lengths
    #
    # # get tpr and fpr values for the average and std of roc curves with different lengths
    # def average_roc_curve(fprs, tprs):
    #     mean_fpr = np.linspace(0, 1, 100)
    #     mean_tpr = np.zeros_like(mean_fpr)
    #     std_tpr = np.zeros_like(mean_fpr)
    #     for fpr, tpr in zip(fprs, tprs):
    #         mean_tpr += np.interp(mean_fpr, fpr, tpr)
    #         std_tpr += np.interp(mean_fpr, fpr, tpr) ** 2
    #     mean_tpr /= len(fprs)
    #     std_tpr = np.sqrt(std_tpr / len(fprs) - mean_tpr ** 2)
    #     return mean_fpr, mean_tpr, std_tpr
    #
    # # get tpr and fpr values for the median and min-max of roc curves with different lengths
    # def median_roc_curve(fprs, tprs):
    #     median_fpr = np.linspace(0, 1, 100)
    #     interpolated_tprs = []
    #     for fpr, tpr in zip(fprs, tprs):
    #         interpolated_tprs.append(np.interp(median_fpr, fpr, tpr))
    #     interpolated_tprs = np.array(interpolated_tprs)
    #
    #     median_tpr = np.median(interpolated_tprs, axis=0)
    #     min_tpr = np.min(interpolated_tprs, axis=0)
    #     max_tpr = np.max(interpolated_tprs, axis=0)
    #     q1_tpr = np.percentile(interpolated_tprs, 25, axis=0)
    #     q3_tpr = np.percentile(interpolated_tprs, 75, axis=0)
    #
    #     return median_fpr, median_tpr, min_tpr, max_tpr, q1_tpr, q3_tpr
    #
    # # make plot with average roc curve and standard deviation
    # _, ax = plt.subplots()
    # mean_fpr, mean_tpr, std_tpr = average_roc_curve(rocs['fpr'], rocs['tpr'])
    # ax.plot(mean_fpr, mean_tpr, color=col, lw=1.5, alpha=.8,
    #         label=f'{classifier} (AUC = {np.mean(aucs):.2f} ± {np.std(aucs):.2f})')
    # ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=col, alpha=.2)
    # ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='black', label='Chance level (AUC = 0.5)', alpha=.8)
    # ax.set_title('ROC curve')
    # ax.set_xlabel('1 - Specificity (FPR)')
    # ax.set_ylabel('Sensitivity (TPR)')
    # ax.set_xlim([-0.05, 1.05])
    # ax.set_ylim([-0.05, 1.05])
    # ax.legend(loc='lower right')
    # plt.grid(alpha=0.3)
    # plt.savefig(os.path.join(args.data_path, results_name+'roc_curve_average.png'), dpi=300, bbox_inches='tight', transparent=True)
    # plt.close()
    #
    # # make plot with median roc curve and min-max
    # _, ax = plt.subplots()
    # median_fpr, median_tpr, min_tpr, max_tpr, q1_tpr, q3_tpr = median_roc_curve(rocs['fpr'], rocs['tpr'])
    # ax.plot(median_fpr, median_tpr, color=col, lw=1.5, alpha=.8,
    #         label=f'{classifier} (AUC = {np.median(aucs):.2f} [{min(aucs):.2f}, {max(aucs):.2f}])')
    # ax.fill_between(median_fpr, min_tpr, max_tpr, color=col, alpha=.2)
    # ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='black', label='Chance level (AUC = 0.5)', alpha=.8)
    # ax.set_title('ROC curve')
    # ax.set_xlabel('1 - Specificity (FPR)')
    # ax.set_ylabel('Sensitivity (TPR)')
    # ax.set_xlim([-0.05, 1.05])
    # ax.set_ylim([-0.05, 1.05])
    # ax.legend(loc='lower right')
    # plt.grid(alpha=0.3)
    # plt.savefig(os.path.join(args.data_path, results_name+'roc_curve_median.png'), dpi=300, bbox_inches='tight', transparent=True)
    # plt.close()
    #
    # # make plot with median roc curve and IQR
    # _, ax = plt.subplots()
    # ax.plot(median_fpr, median_tpr, color=col, lw=1.5, alpha=.8,
    #         label=f'{classifier} (AUC = {np.median(aucs):.2f} [{np.percentile(aucs, 25):.2f}, {np.percentile(aucs, 75):.2f}])')
    # ax.fill_between(median_fpr, q1_tpr, q3_tpr, color=col, alpha=.2)
    # ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='black', label='Chance level (AUC = 0.5)', alpha=.8)
    # ax.set_title('ROC curve')
    # ax.set_xlabel('1 - Specificity (FPR)')
    # ax.set_ylabel('Sensitivity (TPR)')
    # ax.set_xlim([-0.05, 1.05])
    # ax.set_ylim([-0.05, 1.05])
    # ax.legend(loc='lower right')
    # plt.grid(alpha=0.3)
    # plt.savefig(os.path.join(args.data_path, results_name+'roc_curve_iqr.png'), dpi=300, bbox_inches='tight', transparent=True)
    # plt.close()
    #
    # if ext:
    #     # make plot roc curve for external test set
    #     _, ax = plt.subplots()
    #     ax.plot(roc_ext['fpr'][0], roc_ext['tpr'][0], color=col, lw=1.5, alpha=.8,
    #             label=f'{classifier} (AUC = {auc_ext:.2f})')
    #     ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='black', label='Chance level (AUC = 0.5)', alpha=.8)
    #     ax.set_title('ROC curve')
    #     ax.set_xlabel('1 - Specificity (FPR)')
    #     ax.set_ylabel('Sensitivity (TPR)')
    #     ax.set_xlim([-0.05, 1.05])
    #     ax.set_ylim([-0.05, 1.05])
    #     ax.legend(loc='lower right')
    #     plt.grid(alpha=0.3)
    #     plt.savefig(os.path.join(args.data_path, results_name+'roc_curve_external.png'), dpi=300, bbox_inches='tight', transparent=True)
    #     plt.close()
    #
    # # plot confusion matrices in a row
    # fig, axs = plt.subplots(1, len(cms) + 1, figsize=(5 * (len(cms) + 1), 5), sharey='row')
    # plt.rcParams.update({'font.size': 20})
    # fig.text(0.5, 0., 'Predicted labels', ha='center')
    # fig.text(0.1, 0.5, 'True labels', va='center', rotation='vertical')
    # cbar_ax = fig.add_axes([.91, .12, .01, .76])
    # vmin = min([cm.min() for cm in cms])
    # vmax = max([cm.max() for cm in cms])
    # # make space between subplots very small
    # plt.subplots_adjust(wspace=0.02)
    # sns.set(font_scale=2.0)
    # for i, ax in enumerate(axs):
    #     # remove yticks for all but the first plot
    #     if i > 0:
    #         ax.yaxis.set_visible(False)
    #     # set tick label size
    #     ax.tick_params(axis='both', which='major', labelsize=16)
    #     if i == len(cms):
    #         sns.heatmap(cm_ext, annot=True, ax=ax, cmap='Blues', fmt='g', cbar=True,
    #                     cbar_ax=cbar_ax, vmin=vmin, vmax=vmax, annot_kws={"size": 20})
    #         ax.set_title('External Test Set')
    #         continue
    #     ax.set_title(f'Internal Test Set {i + 1}')
    #
    #     sns.heatmap(cms[i], annot=True, ax=ax, cmap='Blues', fmt='g', cbar=True,
    #                 cbar_ax=cbar_ax, vmin=vmin, vmax=vmax, annot_kws={"size": 20})
    # plt.grid(alpha=0.3)
    # plt.savefig(os.path.join(args.data_path, results_name+'confusion_matrices.png'), dpi=300, bbox_inches='tight', transparent=True)
    # plt.close()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)