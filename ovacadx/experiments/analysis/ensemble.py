import os
import argparse
import statistics

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns


def get_args_parser():
    parser = argparse.ArgumentParser(description='Ensemble of nested crossvalidation results')
    parser.add_argument('--data_path', type=str, default='/path/to/results/folder')
    parser.add_argument('--num_inner_splits', type=int, default=5, help='Number of inner splits for nested cross-validation')
    parser.add_argument('--num_outer_splits', type=int, default=5, help='Number of outer splits for nested cross-validation')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
    
    return parser


def main(args):
    # Load results
    results_test = pd.read_csv(os.path.join(args.data_path, 'results_test.csv'))
    results_val = pd.read_csv(os.path.join(args.data_path, 'results_val.csv'))

    # group the test results per outer loop of the nested cross-validation
    mask = results_test.notnull()

    group_identifier = mask.apply(lambda row: tuple(row), axis=1)
    grouped = results_test.groupby(group_identifier)

    # get the thresholds for decision making per outer loop of the nested cross-validation
    # using the validation sets
    best_thresholds = []
    labels = [1 if 'M' in row['tumor_id'].split('_')[1] else 0 for _, row in results_val.iterrows()]
    for column_name, column in results_val.items():
        if not 'output' in column_name:
            continue

        # get column values of all rows and belonging labels without nans
        column_list, labels_list = [], []
        for i, value in column.items():
            if not np.isnan(value):
                column_list.append(torch.sigmoid(torch.tensor(value)).item())
                labels_list.append(labels[i])

        if len(column_list) == 0:
            continue
        
        # calculate the roc curve and find the threshold that maximizes the f1 score
        fpr, tpr, thresholds = roc_curve(labels_list, column_list)
        f1 = 2 * tpr * (1 - fpr) / (tpr + 1 - fpr)
        best_threshold = thresholds[np.argmax(f1)]
        best_thresholds.append(best_threshold)

    print("Thresholds:", best_thresholds)

    aucs = []
    auc_ext = 0.
    rocs = {'fpr': [], 'tpr': []}
    roc_ext = {'fpr': [], 'tpr': []}
    precisions = []
    precisions_ext = 0.
    recalls = []
    recalls_ext = 0.
    avg_precisions = []
    avg_precisions_ext = 0.
    # confusion matrix for every fold
    cms = []
    cm_ext = np.zeros((2, 2))
    # auc_ext = 0.
    for _, group in grouped:
        
        labels = []
        probs = []
        preds = []
        for _, row in group.iterrows():
            label = 1 if 'M' in row['tumor_id'].split('_')[1] else 0
            labels.append(label)
            # take all ther logit values and calculate the mean probability
            # ALSO: apply the threshold to the logit values
            logits = []
            threshlds = []
            clean_row = row.drop('tumor_id')
            for i, (_, value) in enumerate(clean_row.items()):
                if not np.isnan(value):
                    logits.append(value)
                    threshlds.append(best_thresholds[i])
            prob = [torch.sigmoid(torch.tensor(logit)).item() for logit in logits]
            probs.append(prob)
            preds.append([1 if prb > thres else 0 for prb, thres in zip(prob, threshlds)])
        probs = [statistics.mean(prob) for prob in probs]
        #preds = [1 if prob > 0.5 else 0 for prob in probs]
        # majority voting on predictions
        preds = [1 if sum(pred) > len(pred) / 2 else 0 for pred in preds]
        auc = roc_auc_score(labels, probs)
        fpr, tpr, _ = roc_curve(labels, probs)
        avg_precision = average_precision_score(labels, probs)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)

        if not 'AVL' in row['tumor_id']:
            aucs.append(auc)
            rocs['fpr'].append(fpr)
            rocs['tpr'].append(tpr)
            precisions.append(precision)
            recalls.append(recall)
            avg_precisions.append(avg_precision)
            cms.append(np.array([[sum([1 for l, p in zip(labels, preds) if l == 0 and p == 0]), sum([1 for l, p in zip(labels, preds) if l == 0 and p == 1])],
                                 [sum([1 for l, p in zip(labels, preds) if l == 1 and p == 0]), sum([1 for l, p in zip(labels, preds) if l == 1 and p == 1])]]))
        else:
            auc_ext = auc
            roc_ext['fpr'].append(fpr)
            roc_ext['tpr'].append(tpr)
            precisions_ext = precision
            recalls_ext = recall
            avg_precisions_ext = avg_precision
            cm_ext += np.array([[sum([1 for l, p in zip(labels, preds) if l == 0 and p == 0]), sum([1 for l, p in zip(labels, preds) if l == 0 and p == 1])],
                                [sum([1 for l, p in zip(labels, preds) if l == 1 and p == 0]), sum([1 for l, p in zip(labels, preds) if l == 1 and p == 1])]])
    
    print("--------- Internal Test Set ---------")
    print("AUC [IQR] [min - max]:", statistics.median(aucs), "[", np.percentile(aucs, 25), "-", np.percentile(aucs, 75), "]", "[", min(aucs), "-", max(aucs), "]")
    print("Precision [IQR] [min - max]:", statistics.median(precisions), "[", np.percentile(precisions, 25), "-", np.percentile(precisions, 75), "]", "[", min(precisions), "-", max(precisions), "]")
    print("Recall [IQR] [min - max]:", statistics.median(recalls), "[", np.percentile(recalls, 25), "-", np.percentile(recalls, 75), "]", "[", min(recalls), "-", max(recalls), "]")
    print("Average Precision [IQR] [min - max]:", statistics.median(avg_precisions), "[", np.percentile(avg_precisions, 25), "-", np.percentile(avg_precisions, 75), "]", "[", min(avg_precisions), "-", max(avg_precisions), "]")
    print("Specificity [IQR] [min - max]", statistics.median([cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in cms]), "[", np.percentile([cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in cms], 25), "-", np.percentile([cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in cms], 75), "]", "[", min([cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in cms]), "-", max([cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in cms]), "]")
    print("Sensitivity [IQR] [min - max]:", statistics.median([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in cms]), "[", np.percentile([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in cms], 25), "-", np.percentile([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in cms], 75), "]", "[", min([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in cms]), "-", max([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in cms]), "]")
    print("PPV [IQR] [min - max]:", statistics.median([cm[1, 1] / (cm[1, 1] + cm[0, 1]) for cm in cms]), "[", np.percentile([cm[1, 1] / (cm[1, 1] + cm[0, 1]) for cm in cms], 25), "-", np.percentile([cm[1, 1] / (cm[1, 1] + cm[0, 1]) for cm in cms], 75), "]", "[", min([cm[1, 1] / (cm[1, 1] + cm[0, 1]) for cm in cms]), "-", max([cm[1, 1] / (cm[1, 1] + cm[0, 1]) for cm in cms]), "]")
    print("NPV [IQR] [min - max]:", statistics.median([cm[0, 0] / (cm[0, 0] + cm[1, 0]) for cm in cms]), "[", np.percentile([cm[0, 0] / (cm[0, 0] + cm[1, 0]) for cm in cms], 25), "-", np.percentile([cm[0, 0] / (cm[0, 0] + cm[1, 0]) for cm in cms], 75), "]", "[", min([cm[0, 0] / (cm[0, 0] + cm[1, 0]) for cm in cms]), "-", max([cm[0, 0] / (cm[0, 0] + cm[1, 0]) for cm in cms]), "]")

    print("--------- External Test Set ---------")
    print("AUC:", auc_ext)
    print("Precision:", precisions_ext)
    print("Recall:", recalls_ext)
    print("Average Precision:", avg_precisions_ext)
    print("Specificity:", cm_ext[0, 0] / (cm_ext[0, 0] + cm_ext[0, 1]))
    print("Sensitivity:", cm_ext[1, 1] / (cm_ext[1, 0] + cm_ext[1, 1]))
    print("PPV:", cm_ext[1, 1] / (cm_ext[1, 1] + cm_ext[0, 1]))
    print("NPV:", cm_ext[0, 0] / (cm_ext[0, 0] + cm_ext[1, 0]))


    # get tpr and fpr values for the average and std of roc curves with different lengths
    def average_roc_curve(fprs, tprs):
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        std_tpr = np.zeros_like(mean_fpr)
        for fpr, tpr in zip(fprs, tprs):
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            std_tpr += np.interp(mean_fpr, fpr, tpr) ** 2
        mean_tpr /= len(fprs)
        std_tpr = np.sqrt(std_tpr / len(fprs) - mean_tpr ** 2)
        return mean_fpr, mean_tpr, std_tpr
    

    # get tpr and fpr values for the median and min-max of roc curves with different lengths
    def median_roc_curve(fprs, tprs):
        median_fpr = np.linspace(0, 1, 100)
        interpolated_tprs = []
        for fpr, tpr in zip(fprs, tprs):
            interpolated_tprs.append(np.interp(median_fpr, fpr, tpr))
        interpolated_tprs = np.array(interpolated_tprs)
        
        median_tpr = np.median(interpolated_tprs, axis=0)
        min_tpr = np.min(interpolated_tprs, axis=0)
        max_tpr = np.max(interpolated_tprs, axis=0)
        q1_tpr = np.percentile(interpolated_tprs, 25, axis=0)
        q3_tpr = np.percentile(interpolated_tprs, 75, axis=0)
        
        return median_fpr, median_tpr, min_tpr, max_tpr, q1_tpr, q3_tpr
    

    # make plot with average roc curve and standard deviation
    _, ax = plt.subplots()
    mean_fpr, mean_tpr, std_tpr = average_roc_curve(rocs['fpr'], rocs['tpr'])
    ax.plot(mean_fpr, mean_tpr, color='darkorange', lw=1.5, alpha=.8, label=f'CADx model (AUC = {np.mean(aucs):.2f} ± {np.std(aucs):.2f})')
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='darkorange', alpha=.2)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='black', label='Chance level (AUC = 0.5)', alpha=.8)
    ax.set_title('ROC curve - CADx model')
    ax.set_xlabel('1 - Specificity (FPR)')
    ax.set_ylabel('Sensitivity (TPR)')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(args.data_path, 'roc_curve_average.png'), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # make plot with median roc curve and min-max
    _, ax = plt.subplots()
    median_fpr, median_tpr, min_tpr, max_tpr, q1_tpr, q3_tpr = median_roc_curve(rocs['fpr'], rocs['tpr'])
    ax.plot(median_fpr, median_tpr, color='darkorange', lw=1.5, alpha=.8, label=f'CADx model (AUC = {np.median(aucs):.2f} [{min(aucs):.2f}, {max(aucs):.2f}])')
    ax.fill_between(median_fpr, min_tpr, max_tpr, color='darkorange', alpha=.2)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='black', label='Chance level (AUC = 0.5)', alpha=.8)
    ax.set_title('ROC curve - CADx model')
    ax.set_xlabel('1 - Specificity (FPR)')
    ax.set_ylabel('Sensitivity (TPR)')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(args.data_path, 'roc_curve_median.png'), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # make plot with median roc curve and IQR
    _, ax = plt.subplots()
    ax.plot(median_fpr, median_tpr, color='darkorange', lw=1.5, alpha=.8, label=f'CADx model (AUC = {np.median(aucs):.2f} [{np.percentile(aucs, 25):.2f}, {np.percentile(aucs, 75):.2f}])')
    ax.fill_between(median_fpr, q1_tpr, q3_tpr, color='darkorange', alpha=.2)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='black', label='Chance level (AUC = 0.5)', alpha=.8)
    ax.set_title('ROC curve - CADx model')
    ax.set_xlabel('1 - Specificity (FPR)')
    ax.set_ylabel('Sensitivity (TPR)')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(args.data_path, 'roc_curve_iqr.png'), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # make plot roc curve for external test set
    _, ax = plt.subplots()
    ax.plot(roc_ext['fpr'][0], roc_ext['tpr'][0], color='darkorange', lw=1.5, alpha=.8, label=f'CADx model (AUC = {auc_ext:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='black', label='Chance level (AUC = 0.5)', alpha=.8)
    ax.set_title('ROC curve - CADx model')
    ax.set_xlabel('1 - Specificity (FPR)')
    ax.set_ylabel('Sensitivity (TPR)')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(args.data_path, 'roc_curve_external.png'), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


    # plot confusion matrices in a row
    fig, axs = plt.subplots(1, len(cms) + 1, figsize=(5 * (len(cms) + 1), 5), sharey='row')
    plt.rcParams.update({'font.size': 20})
    fig.text(0.5, 0., 'Predicted labels', ha='center')
    fig.text(0.1, 0.5, 'True labels', va='center', rotation='vertical')
    cbar_ax = fig.add_axes([.91,.12,.01,.76])
    vmin = min([cm.min() for cm in cms])
    vmax = max([cm.max() for cm in cms])
    # make space between subplots very small
    plt.subplots_adjust(wspace=0.02)
    sns.set(font_scale=2.0)
    for i, ax in enumerate(axs):
        # remove yticks for all but the first plot
        if i > 0:
            ax.yaxis.set_visible(False)
        ax.xaxis.set_ticklabels(['Benign', 'Malignant'])
        ax.yaxis.set_ticklabels(['Benign', 'Malignant'])
        # set tick label size
        ax.tick_params(axis='both', which='major', labelsize=16)
        if i == len(cms):
            sns.heatmap(cm_ext, annot=True, ax=ax, cmap='Blues', fmt='g', cbar=True,
                        cbar_ax=cbar_ax, vmin=vmin, vmax=vmax, annot_kws={"size": 20})
            ax.set_title('External Test Set')
            continue
        ax.set_title(f'Internal Test Set {i+1}')
            
        sns.heatmap(cms[i], annot=True, ax=ax, cmap='Blues', fmt='g', cbar=True, 
                    cbar_ax=cbar_ax, vmin=vmin, vmax=vmax, annot_kws={"size": 20})
    plt.savefig(os.path.join(args.data_path, 'confusion_matrices.png'), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
