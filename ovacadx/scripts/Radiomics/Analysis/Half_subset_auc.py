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
    for results_test, prefix in zip([results_val,results_test_int],["VAL_"]):
        if prefix == "":
            input("Evaluating on test set. Please only use for debug purposes. Continue?")

        #plt.figure(figsize=(10, 6))
        print("Starting prefix" + prefix)
        if len(results_test)==0: #This happens for LIDC ext, as LIDC does not have an ext dataset.
            print(prefix+" does not hvae samples, skipping")
            continue;
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

        #For this experiment, we don't bootstrap. Instead we simply calculate the AUC over the three groups.



        best_auc_of_all_time = pd.DataFrame({'AUC score': [0,0,0,0]})





        ###
        # The code below can be uncommented to calculate for each feature the maximum half group AUC, and identify features with the highest value.
        # if prefix == "VAL_":
        #     half_group_scores={}
        #     half_group_side = {}
        #     for current_feature_name in tqdm(radiomics.columns[2:], desc="Processing Features"):
        #
        #         #current_feature_name=radiomics.columns[2:][0] #2 is here to skip ID and label
        #
        #         pp_data_int = preprocessed_data[~preprocessed_data['tumor_id'].str.startswith("AVL_")].copy()
        #         pp_data_int = pp_data_int[(pp_data_int['label'] == 0) | (pp_data_int['label'] == 1)]
        #         pp_data_int.loc[:, 'current_feature'] = None
        #
        #         for index, nid in pp_data_int['tumor_id'].items():
        #             matching_row = radiomics[radiomics['ID'] == nid]
        #             if not matching_row.empty:  # Ensure there's a match
        #                 pp_data_int.at[index, 'current_feature'] = matching_row.iloc[0][current_feature_name]
        #
        #
        #
        #         median_volume = np.median(pp_data_int['current_feature'])
        #         # median_volume = np.percentile(pp_data_int['current_feature'],75)
        #         small_group = pp_data_int[pp_data_int['current_feature'] < median_volume]
        #         large_group = pp_data_int[pp_data_int['current_feature'] >= median_volume]
        #         auc = {}
        #
        #         for group, name in zip([pp_data_int,small_group, large_group], ["all", "small", "large"]):
        #             num_mal = len(group[group['label'] == 1])
        #             num_ben = len(group[group['label'] == 0])
        #             if num_ben < 10 or num_mal < 10 or num_ben / num_mal > 10 or num_mal / num_ben > 10:  # Abritrary limts on when AUC is reliable
        #                 print("Cannot compute AUC reliably due to sample imbalance!")
        #             else:
        #                 lbls = []
        #                 preds = []
        #                 for nid in group['tumor_id']:
        #                     row_results_test = results_test[results_test['tumor_id'] == nid]
        #                     preds.append(row_results_test['avg_pred'])
        #                     lbls.append(row_results_test['label'])
        #
        #             auc[name] = roc_auc_score(lbls, preds)
        #         auc_df = pd.DataFrame(list(auc.items()), columns=["Group", "AUC score"])
        #         # Set the value of radiomics[current_feature_name] where the value was "Max_50_AUC" to 0.8
        #         half_group_scores[current_feature_name] = max(auc_df['AUC score'][1:]) #We take 1: as we want to skip the score for "ALL"
        #         half_group_side[current_feature_name] =np.argmax(auc_df['AUC score'][1:]) #0 for below threshold, 1 for above threshold
        #         if max(auc_df['AUC score']) > max(best_auc_of_all_time['AUC score']):
        #             best_auc_of_all_time = auc_df
        #             best_feature_of_all_time = current_feature_name
        #     print(half_group_scores)
        #     save_dict_to_file(half_group_side, 'half_group_side.pkl')
        #     save_dict_to_file(half_group_scores, 'half_group_scores.pkl')




        half_group_side = load_dict_from_file('half_group_side.pkl')
        half_group_scores = load_dict_from_file('half_group_scores.pkl')


        # Sort the dictionary by value in descending order
        sorted_items = sorted(half_group_scores.items(), key=lambda item: item[1], reverse=True)

        sorted_items_df = pd.DataFrame(sorted_items, columns=['Feature name', 'Optimal AUC @ 50%'])
        sorted_items_df['Above threshold'] = sorted_items_df['Feature name'].map(half_group_side).fillna(0) #0 means subset below threshold, 1 means above threshold
        print(sorted_items_df.head(10))
        sorted_items_df.to_csv("Optimal_half_AUCs_per_feature.csv")
        # Initialize a plot
        cols = plt.rcParams['axes.prop_cycle'].by_key()['color'] #retrieves all colors
        # del cols[5] #5 is brown is ugly
        i=0


        # #make sure that first feature is volume
        row_to_move = sorted_items_df[sorted_items_df['Feature name'] == 'original_shape_VoxelVolume']  # Move the row where column 'A' has value 3
        # Remove the row from the original DataFrame and concatenate it at the top
        sorted_items_df = pd.concat([row_to_move, sorted_items_df[sorted_items_df['Feature name'] != 'original_shape_VoxelVolume']], ignore_index=True)
        #
        plot_labels=[]
        AUC_curves_per_feature={}
        # Get the top 5 keys and iterate through them
        indices = [0, 200, 420, 1050, -20] #Last value doesnt matter
        # indices = [0,200,400,600,800]
        for index, row in sorted_items_df.iloc[indices].iterrows(): #:x is the first x features (0 and 1 are duplicates in our case)

            if i ==4:
                key = list(sorted_items_df['Feature name'])[-5]

            else:
                key = row['Feature name']
            print(key)
            # value = row['Optimal AUC @ 50%']
            current_feature_name = key
            # Subset creation and AUC calculation
            N = 75 #Resolution for plot
            thresholds = np.linspace(1/N, 1.0, N)  # Threshold percentages (10%, 20%, ..., 100%)
            x_percentages = []  # X-axis values
            aucs={}
            # Filter the dataset to include only benign and malignant samples
            pp_data_int = preprocessed_data[~preprocessed_data['tumor_id'].str.startswith("AVL_")].copy()
            pp_data_int = pp_data_int[(pp_data_int['label'] == 0) | (pp_data_int['label'] == 1)]
            pp_data_int.loc[:, 'current_feature'] = None
            median_volume = np.median(pp_data_int['volume_cm3'])
            #
            for index, nid in pp_data_int['tumor_id'].items(): #Sets pp_data_int['current_feature']
                matching_row = radiomics[radiomics['ID'] == nid]
                if not matching_row.empty:  # Ensure there's a match
                    pp_data_int.at[index, 'current_feature'] = matching_row.iloc[0][current_feature_name]
            for percent in thresholds:
                # percent = 0.3
                if half_group_side[current_feature_name] > 0: #Above threshold
                    threshold_volume = np.percentile(pp_data_int['current_feature'],100-percent*100)
                    group = pp_data_int[pp_data_int['current_feature'] >= threshold_volume]
                else: #Below threshold
                    threshold_volume = np.percentile(pp_data_int['current_feature'], percent * 100)
                    group = pp_data_int[pp_data_int['current_feature'] <= threshold_volume]


                # threshold_volume = np.percentile(pp_data_int['current_feature'],percent*100)
                # small_group = pp_data_int[pp_data_int['current_feature'] <= threshold_volume]
                # large_group = pp_data_int[pp_data_int['current_feature'] >= threshold_volume]
                # auc = {}
                #
                # if half_group_side[current_feature_name]>0:
                #     group=large_group
                # else:
                #     group=small_group

                num_mal = len(group[group['label'] == 1])
                num_ben = len(group[group['label'] == 0])
                if num_ben < 10 or num_mal < 10 or num_ben / num_mal > 10 or num_mal / num_ben > 10:  # Abritrary limts on when AUC is reliable
                    # print("Cannot compute AUC reliably due to sample imbalance!",current_feature_name,threshold_volume,num_ben, num_mal)
                    # aucs[percent] = 0 <-- Will set uncalculatable options to 0
                    pass
                else:
                    lbls = []
                    preds = []
                    for nid in group['tumor_id']:
                        row_results_test = results_test[results_test['tumor_id'] == nid]
                        preds.append(row_results_test['avg_pred'])
                        lbls.append(row_results_test['label'])
                    aucs[percent] = roc_auc_score(lbls, preds)
                    # print("c")

                # Store the percentage of samples in the subset for X-axis
                x_percentages.append(percent)#len(group_below) / len(pp_data_int) * 100)


            # if half_group_side[current_feature_name] ==1:
            #     #Reverse AUCs as to fix x-axis for samples that are ABOVE threshold
            #     aucs = {1 - k: v for k, v in aucs.items()}
            aucs_df = pd.DataFrame(list(aucs.items()), columns=['Key', 'Value'])
            filtered_dict = {k: v for k, v in aucs.items() if 0.2 <= k <= 1.0} #plot from 0.2 to 1.0

            x = list(filtered_dict.keys())
            #x = list(1-b for b in x)
            y = list(filtered_dict.values())
            AUC_curves_per_feature[current_feature_name] = (x[::-1], y) #invert x so that the x-axis is increasing rather than decreasing
            # print("SET_",current_feature_name)
            i+=1
    #plt.plot(x_percentages, auc_above, label="AUC for Above Threshold", marker='o')
        AUC_curves[prefix] = AUC_curves_per_feature

    # import matplotlib.pyplot as plt
    # import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(10*0.9, 3.8*0.9))
    ax = axes[0]
    values = list(half_group_scores.values())
    sns.histplot(values, stat="density", kde=False, bins=40, color='lightsteelblue', ax=ax, kde_kws={'color':'blue','bw_adjust': 1.5})
    # sns.kdeplot(values, color='blue', bw_adjust=1.5, ax=ax)
    ax.grid(True)
    ax.set_title(r'Distribution of AUC$_{\mathrm{median}}$ on the val set') # fontsize=14, weight='bold'
    ax.set_xlabel(r'AUC$_{\mathrm{median}}$') #, fontsize=12
    ax.set_ylabel('Number of features') # fontsize=12
    a = AUC_curves['VAL_'] #All curves here should be on VAL set! Because we are determining a hyperparameter (T)
    F=[]
    for key in a.keys():
        F.append(sorted_items_df[sorted_items_df['Feature name']==key]['Optimal AUC @ 50%'])
    # Highlight specific values with vertical lines
    # add baseline line
    ax.axvline(x=0.815, color="black", linestyle='dotted', linewidth=2, label=f'BL') #0.815 is the baseline score of the MILCNN + R model
    for i, f in enumerate(F):
        ax.axvline(x=f.iloc[0], color=cols[i], linestyle='dashdot', linewidth=2, label=f'$f_{i + 1}$')


    # Add legend
    ax.text(-0.02, 1, chr(65 + 0), transform=ax.transAxes, fontsize=14, weight='bold',
            verticalalignment='bottom', horizontalalignment='right', fontname='Nimbus Roman')

    ax.legend(loc='upper center', bbox_to_anchor=(0.16, 1.0),
              fancybox=False, shadow=False)

    # Adjust axis limits
    # ax.set_xlim(min(values) - 0.5, max(values) + 0.5)




    ax = axes[1]
    i=0
    for current_feature_name in a.keys():
        x,y = a[current_feature_name]
        # x[::-1] Inverts the x axis
        # if prefix == "":
        #     ax.plot(x[::-1], y, label="F" + str(i), marker='x', color=cols[i])
        # else:

        if i == 0:  # Check if it's the first feature
            first_x = x[0]  # Get the first datapoint after inversion
            first_y = y[::-1][0]  # Corresponding y value
            ax.plot(first_x, first_y, marker='*', color='white',markeredgewidth=1, markeredgecolor='black', markersize=14, label="BL",zorder=5)

        ax.plot(x[::-1], y, label="$f_" + str(i+1)+"$", marker='^', color=cols[i],markersize=4)
        plot_labels.append(str(current_feature_name))
        # if prefix == "VAL_":
        #     title_name = "Subset size vs AUC on val data"
        # else:
        title_name = "Subset size vs subset AUC on val set"
        ax.set_title(title_name)
        ax.legend(ncols=2)
        ax.set_xlabel("Proportional subset size ($T$)")
        ax.set_ylabel("Subset AUC Score")
        ax.invert_xaxis()
        # ax.invert_xaxis()
        ax.grid(True)
        ax.set_ylim(0.7, 0.98)
        ax.text(-0.02, 1, chr(65 + 1), transform=ax.transAxes, fontsize=14, weight='bold',
                verticalalignment='bottom', horizontalalignment='right', fontname='Nimbus Roman')
        i+=1
    # ax.show()
    # plt.tight_layout()
    # plt.show()

    #This code plots val on left, test on right
    # fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    # for results_test, prefix, ax in zip([results_val, results_test_int], ["VAL_", ""], axes):
    #     a = AUC_curves[prefix]
    #     i=0
    #     plot_labels=[]
    #     for current_feature_name in a.keys():
    #         x,y = a[current_feature_name]
    #
    #         # x[::-1] Inverts the x axis
    #         if prefix == "":
    #             ax.plot(x[::-1], y, label="F" + str(i), marker='x', color=cols[i])
    #         else:
    #             ax.plot(x[::-1], y, label="F" + str(i), marker='o', color=cols[i])
    #         plot_labels.append(str(current_feature_name))
    #         if prefix == "VAL_":
    #             title_name = "Subset size vs AUC on val data"
    #         else:
    #             title_name = "Subset size vs AUC on test data"
    #         ax.set_title(title_name)
    #         ax.legend()
    #         ax.set_xlabel("Percentage of Samples in Subset (%)")
    #         ax.set_ylabel("AUC Score")
    #         # ax.invert_xaxis()
    #         ax.grid(True)
    #         ax.set_ylim(0.78, 0.98)
    #         i+=1


    # fig.set_title("Subset Size vs AUC after ranking tumors by feature")
    plt.tight_layout()
    plt.savefig("Subset_aucs.pdf")
    plt.show()
    print(plot_labels)


############################################################################################
    #Fetch half group score for the TEST set for Volume feature
    results_test = results_test_int
    half_group_scores = {}
    half_group_side = {}
    current_feature_name = "original_shape_VoxelVolume"
    #Average the test set results of outer loop of nested cross validation.
    for ind, row in results_test.iterrows():
        lbl = int(preprocessed_data[preprocessed_data['tumor_id'] == row['tumor_id']]['label'].iloc[0])
        results_test.at[ind, 'label'] = lbl
        logits = []
        clean_row = row.drop('tumor_id')
        for i, (_, value) in enumerate(clean_row.items()):
            if not np.isnan(value):
                logits.append(value)
        results_test.at[ind, 'avg_pred'] = statistics.mean(logits)

    #current_feature_name=radiomics.columns[2:][0] #2 is here to skip ID and label
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
    for group, name in zip([pp_data_int,small_group, large_group], ["all", "small", "large"]):
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
    half_group_scores[current_feature_name] = max(auc_df['AUC score'][1:]) #We take 1: as we want to skip the score for "ALL"
    half_group_side[current_feature_name] =np.argmax(auc_df['AUC score'][1:]) #0 for below threshold, 1 for above threshold
    if max(auc_df['AUC score']) > max(best_auc_of_all_time['AUC score']):
        best_auc_of_all_time = auc_df
        best_feature_of_all_time = current_feature_name
    print(half_group_scores)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)