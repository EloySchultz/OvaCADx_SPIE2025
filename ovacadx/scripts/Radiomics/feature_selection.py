# LOAD FEATURES

from tqdm import tqdm
import os
import argparse
import json
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wasserstein_distance
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
from scipy.stats import pearsonr, spearmanr, shapiro
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from classifiers import train_classifiers, check_disjoint, save_models, load_models
from utils import filter_df, get_dataset
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
def get_args_parser():
    parser = argparse.ArgumentParser(description='Vizualize radiomics')
    parser.add_argument('--data_path', type=str,
                        default='/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/')  # default='/path/to/output/folder', help='Path to output folder')
    return parser



def df_to_array(df):
    """
    Converts Dataframe to numpy array, extracting the feature names, classes and IDs
    Args:
        df:

    Returns: NumpyArray (num_features x num_samples), list (feature_names), list (classes), list(IDs)

    """
    IDs = list(df['ID'])
    classes = list(df['label'])
    df = df.drop(columns=['ID'])
    df = df.drop(columns=['label'])
    return df.to_numpy(), list(df.columns),classes,IDs


def array_to_df(radiomicFeatures3D, feature_names, classes,IDs): #r = radiomics_array object
    """
    Inverse of df_to_array, so converts array back into a dataframe.
    Args:
        radiomicFeatures3D: (numpy array containing features, num_features x num_samples)
        feature_names: list containing names of features
        classes: list of classes (same order as IDs)
        IDs: list of IDs

    Returns: dataframe

    """
    df = pd.DataFrame(radiomicFeatures3D, columns=feature_names)
    df.insert(0, 'label', classes)
    df.insert(0, 'ID', IDs)
    return df


def feature_selection_pca(df, num_features):
    """
    Applies PCA for feature selection on the given dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe with ID, label, and features.
    - label_column (str): The name of the label column.
    - num_features (int): The number of principal components to retain.

    Returns:
    - pd.DataFrame: DataFrame with ID, label, and selected principal components.
    """
    label_column = 'label'
    # Extracting IDs and labels
    radiomicFeatures3D, feature_names, classes, IDs = df_to_array(df)

    # Dropping the ID and label columns to get the feature columns
    #feature_columns = df.drop(columns=['ID', label_column])
    feature_columns = radiomicFeatures3D
    # Applying PCA
    pca = PCA(n_components=num_features)
    principal_components = pca.fit_transform(feature_columns)
    PC_names = [f'PC{i + 1}' for i in range(num_features)]
    # Creating a DataFrame for the principal components
    principal_df = pd.DataFrame(data=principal_components,
                                columns=PC_names)

    principal_df = array_to_df(principal_df,PC_names,classes,IDs)


    return principal_df, pca


def feature_selection_select_percentile(df, percentile):
    # Define the column name for the label
    label_col = 'label'

    # Split the DataFrame into features (X) and the target variable (y)
    X = df.drop(columns=['ID', label_col])
    y = df[label_col]

    # Initialize the SelectPercentile object
    selector = SelectPercentile(score_func=f_classif, percentile=percentile)

    # Fit the selector to the data
    X_selected = selector.fit_transform(X, y)

    # Get the selected feature names
    selected_features = X.columns[selector.get_support()]

    # Create a new DataFrame with only the selected features
    df_selected = pd.DataFrame(X_selected, columns=selected_features)

    # Add the ID and label columns back to the DataFrame
    df_selected['ID'] = df['ID'].values
    df_selected[label_col] = y.values

    return df_selected



def feature_selection_ps(df, label_col='label', percentile=np.inf, num_features=np.inf):
    """
    Perform feature selection using SelectPercentile and correlation analysis.

    Parameters:
    - df: pandas DataFrame, the input data with features, ID and label columns.
    - label_col: str, the name of the label column.
    - percentile: int, the percentage of top features to select using SelectPercentile.

    Returns:
    - df_selected: pandas DataFrame, the DataFrame with the selected features retained.
    """
    if num_features==np.inf:
        num_features = min(1,int(len(df)*percentile/100))
    # Separate features and labels
    X = df.drop(columns=['ID', label_col])
    y = df[label_col]
    # Step 2: Correlation analysis
    def is_normal_distribution(series):
        """Check if a series follows a normal distribution using the Shapiro-Wilk test."""
        stat, p_value = shapiro(series)
        return p_value > 0.05

    # Assuming X is already defined as your DataFrame
    feature_names = list(X.columns)

    #For speaman vs pearson normality, see above:
    #https://stats.stackexchange.com/questions/3730/pearsons-or-spearmans-correlation-with-non-normal-data \
    selected_features=[]
    # Calculate the correlation matrix
    corr_matrix_pearson = X.corr(method='pearson')
    corr_matrix_spearman = X.corr(method='spearman')
    corr_matrix_all = corr_matrix_spearman.copy()
    corr_matrix_pearson=corr_matrix_pearson.abs()
    corr_matrix_spearman=corr_matrix_spearman.abs()
    # np.fill_diagonal(corr_matrix_spearman.values, 0)
    # np.fill_diagonal(corr_matrix_pearson.values, 0) DIAGOONALS

    # Step 1: Determine which features follow a normal distribution
    is_normal = np.array([is_normal_distribution(X[i]) for i in X.columns])

    # Step 2: Create the correlation matrix
    corr_matrix_all_np = np.where(np.outer(is_normal, is_normal), corr_matrix_pearson, corr_matrix_spearman)
    corr_matrix_all = pd.DataFrame(corr_matrix_all_np, index=X.columns, columns=X.columns)

    #Start by picking the feature with least correlation
    total_correlation_per_feature = corr_matrix_all_np.sum(axis=1)
    total_correlation_df = pd.DataFrame({
        'feature_names': feature_names,
        'total_correlation': total_correlation_per_feature
    })
    total_correlation_df_sorted = total_correlation_df.sort_values(by='total_correlation', ascending=True)
    selected_features.append(total_correlation_df_sorted.head(1)['feature_names'].values[0])

    corr_matrix_all=corr_matrix_all.drop(selected_features[-1])# Drop row of selected featyure
    corr_matrix_all_np = np.array(corr_matrix_all)
    feature_names.remove(selected_features[-1])
    for N in range(num_features):
        mask_matrix = corr_matrix_all.copy()
        # Step 2: Set the entire matrix to zero
        mask_matrix.iloc[:, :] = 0
        # Step 3: Set the columns of selected features to 1
        for feature in selected_features:
            mask_matrix[feature] = 1

        corr_matrix = corr_matrix_all*mask_matrix #Only columns of selected features will be 1. Hence, we find a feature that has least average correlation with all features selected so far.
        total_correlation_per_feature = corr_matrix.sum(axis=1)
        total_correlation_df = pd.DataFrame({
            'feature_names': feature_names,
            'total_correlation': total_correlation_per_feature
        })
        total_correlation_df_sorted = total_correlation_df.sort_values(by='total_correlation', ascending=True)
        selected_features.append(total_correlation_df_sorted.head(1)['feature_names'].values[0])

        corr_matrix_all = corr_matrix_all.drop(selected_features[-1])  # Drop row of selected featyure
        corr_matrix_all_np = np.array(corr_matrix_all)
        feature_names.remove(selected_features[-1])
    selected_features.append('label')
    selected_features.append('ID')
    selected_final_features_df = df[selected_features]

    return selected_final_features_df

from kneed import KneeLocator
import numpy as np


def linear_kneepoint_selection(x,y):

    x=np.array(x)
    # Line from the first to the last point
    # Line from the first to the last point
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    # Slope and intercept of the line
    slope = (y2 - y1) / (x2 - x1)
    if slope < 0:
        slope =0 #
        intercept = (y1 + y2)/2 #Simply horizontal line through center of graph, i.e. take maximum
    else:
        intercept = y1 - slope * x1

    # Calculate the y value on the trend line for each x
    trend_line_y = slope * x + intercept

    # Calculate perpendicular distance only for points above or on the line
    distances = np.where(y >= trend_line_y, np.abs(slope * x - y + intercept) / np.sqrt(slope ** 2 + 1), 0)

    # Find the index of the maximum distance
    knee_index = np.argmax(distances)

    # Return the x coordinate of the knee point
    return x[knee_index]


def kneedle_kneepoint_selection(adjusted_num_features,smoothed_accuracy_values,auto_select_kp):
    knee_point_accuracy = None
    knee_point_auc = None
    if auto_select_kp == "ACC":
        # Find the knee point in the smoothed accuracy vs. number of features graph

        kneedle = KneeLocator(adjusted_num_features, smoothed_accuracy_values, curve='concave',
                              direction='decreasing')
        knee_point_accuracy = kneedle.knee
        num_features_to_keep = knee_point_accuracy
        if num_features_to_keep is None:  # If no knee point is found, simply take half features.
            num_features_to_keep = len(adjusted_num_features) // 2

    # elif auto_select_kp == "AUC":
    #     # Find the knee point in the smoothed AUC vs. number of features graph
    #     kneedle_auc = KneeLocator(adjusted_num_features, smoothed_auc_values, curve='concave',
    #                               direction='decreasing')
    #     knee_point_auc = kneedle_auc.knee
    #     num_features_to_keep = knee_point_auc
    #     if num_features_to_keep is None:  # If no knee point is found, simply take half features.
    #         num_features_to_keep = len(num_features) // 2
    return num_features_to_keep
def feature_selection_black_box_rf(df, mod, label_col='label', auto_select_kp = "ACC", show_graph=True):
    """
    Perform feature selection using a wrapper method based on Random Forest classifier and plot accuracy & AUC vs. number of features.

    Parameters:
    - df: pandas DataFrame, the input data with features, ID and label columns.
    - csv_path: str, the path to the CSV file containing the dataset for obtaining folds.
    - label_col: str, the name of the label column.
    - auto_select_kp, What metric to use for automatic kneepoint selection. Either "ACC", "AUC" or None. If None, manual knee point selection is enabled (requires human input)

    Returns:
    - df_selected: pandas DataFrame, the DataFrame with the selected features retained.
    """

    # Separate features and labels
    X = df.drop(columns=['ID', label_col])
    y = df[label_col]
    feature_list = X.columns.tolist()

    # Placeholder for accuracies and AUCs
    accuracies = []
    aucs = []
    #print("Training feature ablation on 5-fold RF classifier")

    # Obtain train and validation sets for the current fold
    #mod = get_dataset(k, csv_path,seed,remove_outliers)
    train_df = filter_df(df, mod.train_samples)
    val_df = filter_df(df, mod.val_samples)

    X_train = train_df.drop(columns=['ID', label_col])
    y_train = train_df[label_col]
    X_val = val_df.drop(columns=['ID', label_col])
    y_val = val_df[label_col]

    # Initialize RF classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Get feature importances
    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)

    # Iteratively remove features with least importance and calculate accuracy and AUC
    fold_accuracies = []
    fold_aucs = []
    for i in tqdm(range(len(sorted_indices))):
        selected_features = sorted_indices[i:]
        X_train_selected = X_train.iloc[:, selected_features]
        X_val_selected = X_val.iloc[:, selected_features]

        rf.fit(X_train_selected, y_train)
        y_pred = rf.predict(X_val_selected)
        y_pred_prob = rf.predict_proba(X_val_selected)[:, 1]
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_prob)
        fold_accuracies.append((len(selected_features), accuracy))
        fold_aucs.append((len(selected_features), auc))

    accuracies.append(fold_accuracies)
    aucs.append(fold_aucs)

    # # Average the accuracies and AUCs over the folds
    # avg_accuracies = []
    # avg_aucs = []
    # for i in range(len(feature_list)):
    #     avg_accuracy = np.mean([fold_acc[i][1] for fold_acc in accuracies])
    #     avg_auc = np.mean([fold_auc[i][1] for fold_auc in aucs])
    #     avg_accuracies.append((len(feature_list) - i, avg_accuracy))
    #     avg_aucs.append((len(feature_list) - i, avg_auc))
    #
    # # Plot accuracy vs. number of features
    # num_features = [acc[0] for acc in avg_accuracies]
    # avg_accuracy_values = [acc[1] for acc in avg_accuracies]
    #
    # plt.figure(figsize=(10, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(num_features, avg_accuracy_values, marker='o', color='b')
    # plt.xlabel('Number of Features')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs. Number of Features')
    # plt.grid(True)
    #
    # # Plot AUC vs. number of features
    # avg_auc_values = [auc[1] for auc in avg_aucs]
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(num_features, avg_auc_values, marker='o', color='r')
    # plt.xlabel('Number of Features')
    # plt.ylabel('AUC')
    # plt.title('AUC vs. Number of Features')
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show()
    # Define a simple moving average function
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')



    # Average the accuracies and AUCs over the folds
    avg_accuracies = []
    avg_aucs = []
    for i in range(len(feature_list)):
        avg_accuracy = np.mean([fold_acc[i][1] for fold_acc in accuracies])
        avg_auc = np.mean([fold_auc[i][1] for fold_auc in aucs])
        avg_accuracies.append((len(feature_list) - i, avg_accuracy))
        avg_aucs.append((len(feature_list) - i, avg_auc))

    if auto_select_kp is not None:

        num_features_to_keep = 0
        while (num_features_to_keep < 5):  # The number of features to keep must be larger than 5
            # AUTOMATIC KNEE POINT SELECTION BELOW
            # Extract the number of features and average accuracy values
            num_features = [acc[0] for acc in avg_accuracies]
            avg_accuracy_values = [acc[1] for acc in avg_accuracies]

            # Extract the average AUC values
            avg_auc_values = [auc[1] for auc in avg_aucs]

            # Apply moving average to smooth the accuracy and AUC values
            window_size = 5
            smoothed_accuracy_values = moving_average(avg_accuracy_values, window_size)
            smoothed_auc_values = moving_average(avg_auc_values, window_size)

            # Adjust the number of features to match the length of the smoothed data
            adjusted_num_features = num_features[window_size - 1:]

            #num_features_to_keep = kneedle_kneepoint_selection(adjusted_num_features, smoothed_accuracy_values, auto_select_kp)
            num_features_to_keep = linear_kneepoint_selection(adjusted_num_features,smoothed_accuracy_values)
            if num_features_to_keep < 5:  # Remove the lowest entries from the list, as we always want more than 5 features
                min_tuple = min(avg_accuracies, key=lambda x: x[0])
                avg_accuracies.remove(min_tuple)
                min_tuple = min(avg_aucs, key=lambda x: x[0])
                avg_aucs.remove(min_tuple)

        #print("Auto-selected: " + str(num_features_to_keep) +" features in black box RF method")
    else:
        show_graph=True #Always show graph for manual optimum point selection
    if show_graph:
        # Plot accuracy vs. number of features with knee point
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(num_features, avg_accuracy_values, marker='o', color='b', alpha=0.3, label='Original')
        plt.plot(adjusted_num_features, smoothed_accuracy_values, marker='o', color='b', label='Smoothed')
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Number of Features')
        plt.legend()
        plt.grid(True)

        # Highlight the knee point in the accuracy graph
        if False:
            plt.plot(num_features_to_keep, smoothed_accuracy_values[adjusted_num_features == num_features_to_keep], 'r*', markersize=15, label='Knee Point')
            plt.legend()

        # Plot AUC vs. number of features with knee point
        plt.subplot(1, 2, 2)
        plt.plot(num_features, avg_auc_values, marker='o', color='r', alpha=0.3, label='Original')
        plt.plot(adjusted_num_features, smoothed_auc_values, marker='o', color='r', label='Smoothed')
        plt.xlabel('Number of Features')
        plt.ylabel('AUC')
        plt.title('AUC vs. Number of Features')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        #Another plot, but now with  knee point
        # Plot accuracy vs. number of features with knee point
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(num_features, avg_accuracy_values, marker='o', color='b', alpha=0.3, label='Original')
        plt.plot(adjusted_num_features, smoothed_accuracy_values, marker='o', color='b', label='Smoothed')
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Number of Features')
        plt.grid(True)

        # Highlight the knee point in the accuracy graph
        if True:
            plt.plot(num_features_to_keep, smoothed_accuracy_values[adjusted_num_features == num_features_to_keep],
                     'r*', markersize=15, label='Knee Point')
            plt.legend()

        # Plot AUC vs. number of features with knee point
        plt.subplot(1, 2, 2)
        plt.plot(num_features, avg_auc_values, marker='o', color='r', alpha=0.3, label='Original')
        plt.plot(adjusted_num_features, smoothed_auc_values, marker='o', color='r', label='Smoothed')
        plt.xlabel('Number of Features')
        plt.ylabel('AUC')
        plt.title('AUC vs. Number of Features')
        plt.grid(True)

        # Highlight the knee point in the AUC graph
        # if knee_point_auc:
        #     plt.plot(knee_point_auc, kneedle_auc.knee_y, 'r*', markersize=15, label='Knee Point')
        #     plt.legend()

        plt.tight_layout()
        plt.show()
    if auto_select_kp is None:
        # Prompt user to select the number of features to keep
        num_features_to_keep = int(input(f"Select the number of features to keep (1 to {len(feature_list)}): "))

    # Get the final selected features
    selected_final_features = [feature_list[idx] for idx in sorted_indices[-num_features_to_keep:]]

    # Return the DataFrame with selected features
    df_selected = pd.concat([df[['ID', label_col]], X[selected_final_features]], axis=1)

    return df_selected, num_features_to_keep


def feature_selection_lda(df, n_components_pca=1,num_features=10):
    n_components_pca = num_features-1
    """
    Apply LDA and PCA for feature selection and visualization to a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'ID', 'label', and features.
    n_components_pca (int): The number of PCA components to reduce to (typically 2 for visualization).

    Returns:
    pd.DataFrame: The DataFrame with the LDA and PCA selected features.
    """
    label_col = 'label'

    # Separate features and labels
    IDs = df['ID']
    features = df.drop(columns=[label_col, 'ID'])
    labels = df[label_col]

    # Apply LDA
    lda = LDA(n_components=1)  # n_components is 1 because of binary classification
    features_lda = lda.fit_transform(features, labels)

    # Combine LDA component with original features
    combined_features = pd.DataFrame(features_lda, columns=['LDA1'])
    combined_features = pd.concat([combined_features, features.reset_index(drop=True)], axis=1)

    # Apply PCA to the combined features
    pca = PCA(n_components=n_components_pca)
    features_pca = pca.fit_transform(combined_features)

    # Create a DataFrame with the selected features
    selected_features_df = pd.DataFrame(features_pca, columns=[f'PC{i + 1}' for i in range(n_components_pca)])

    # Add ID and label columns back to the filtered DataFrame
    selected_features_df.insert(0, 'ID', IDs)
    selected_features_df.insert(1, 'label', labels)

    return selected_features_df

def feature_selection_anova(df, num_features):
    """
    Apply ANOVA feature selection to a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'ID', 'label', and features.
    label_col (str): The name of the column containing the label.
    num_features (int): The number of top features to select.

    Returns:
    pd.DataFrame: The DataFrame with the top 'num_features' selected.
    """
    label_col = 'label'
    # Separate features and labels
    IDs = df['ID']

    features = df.drop(columns=[label_col, 'ID'])
    labels = df[label_col]
    classes=labels
    # Apply SelectKBest with ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=num_features)
    selector.fit(features, labels)

    # Get the columns to keep
    cols = selector.get_support(indices=True)
    selected_features = features.columns[cols]

    # Add label column back to the filtered DataFrame
    df = df[selected_features.to_list()]
    df.insert(0, 'label', classes)
    df.insert(0, 'ID', IDs)
    df=df.copy()

    return df


# def feature_selection_ksps(df,num_features,subset_list,alpha):
#     #High alpha = more emphasis on correlation prevention. Low alha == more emphasis on KS similarity
#     label_col = 'label'
#     # Separate features and labels
#     X = df.drop(columns=['ID', label_col])
#     X_ks = X.copy()
#     y = df[label_col]
#     # Step 2: Correlation analysis
#     def is_normal_distribution(series):
#         """Check if a series follows a normal distribution using the Shapiro-Wilk test."""
#         stat, p_value = shapiro(series)
#         return p_value > 0.05
#
#     # Assuming X is already defined as your DataFrame
#     feature_names = list(X.columns)
#
#     #For speaman vs pearson normality, see above:
#     #https://stats.stackexchange.com/questions/3730/pearsons-or-spearmans-correlation-with-non-normal-data \
#     selected_features=[]
#     # Calculate the correlation matrix
#     corr_matrix_pearson = X.corr(method='pearson')
#     corr_matrix_spearman = X.corr(method='spearman')
#     corr_matrix_all = corr_matrix_spearman.copy()
#     corr_matrix_pearson=corr_matrix_pearson.abs()
#     corr_matrix_spearman=corr_matrix_spearman.abs()
#     # np.fill_diagonal(corr_matrix_spearman.values, 0)
#     # np.fill_diagonal(corr_matrix_pearson.values, 0) DIAGOONALS
#
#     # Step 1: Determine which features follow a normal distribution
#     is_normal = np.array([is_normal_distribution(X[i]) for i in X.columns])
#
#     # Step 2: Create the correlation matrix
#     corr_matrix_all_np = np.where(np.outer(is_normal, is_normal), corr_matrix_pearson, corr_matrix_spearman)
#     corr_matrix_all = pd.DataFrame(corr_matrix_all_np, index=X.columns, columns=X.columns)
#
#     #Start by picking the feature with least correlation
#     total_correlation_per_feature = corr_matrix_all_np.sum(axis=1)
#     def normalize(x):
#         x = (x-np.min(x))/(np.max(x)-np.min(x))
#         return np.array(x)
#
#     #ADD KS
#     ks_similarities = get_ks_similarity_interclass(np.array(X_ks),subset_list)
#     total_correlation_df = pd.DataFrame({
#         'feature_names': feature_names,
#         'total_correlation': alpha*normalize(total_correlation_per_feature) + (1-alpha)*normalize(ks_similarities)
#     })
#     total_correlation_df_sorted = total_correlation_df.sort_values(by='total_correlation', ascending=True)
#     selected_features.append(total_correlation_df_sorted.head(1)['feature_names'].values[0])
#
#     corr_matrix_all=corr_matrix_all.drop(selected_features[-1])# Drop row of selected featyure
#     corr_matrix_all_np = np.array(corr_matrix_all)
#     X_ks = np.delete(X_ks, feature_names.index(selected_features[-1]),axis=1)
#     feature_names.remove(selected_features[-1])
#     for N in range(num_features):
#         mask_matrix = corr_matrix_all.copy()
#         # Step 2: Set the entire matrix to zero
#         mask_matrix.iloc[:, :] = 0
#         # Step 3: Set the columns of selected features to 1
#         for feature in selected_features:
#             mask_matrix[feature] = 1
#
#         corr_matrix = corr_matrix_all*mask_matrix
#         total_correlation_per_feature = corr_matrix.sum(axis=1)
#         ks_similarities = get_ks_similarity_interclass(np.array(X_ks), subset_list)
#         total_correlation_df = pd.DataFrame({
#             'feature_names': feature_names,
#             'total_correlation':alpha*normalize(total_correlation_per_feature) + (1-alpha)*normalize(ks_similarities)
#         })
#         total_correlation_df_sorted = total_correlation_df.sort_values(by='total_correlation', ascending=True)
#         selected_features.append(total_correlation_df_sorted.head(1)['feature_names'].values[0])
#
#         corr_matrix_all = corr_matrix_all.drop(selected_features[-1])  # Drop row of selected featyure
#         corr_matrix_all_np = np.array(corr_matrix_all)
#         X_ks=np.delete(X_ks,feature_names.index(selected_features[-1]),axis=1)
#         feature_names.remove(selected_features[-1])
#
#
#     selected_final_features_df = df[selected_features]
#
#     return selected_final_features_df



def feature_selection_ksps():
    #Old function, here so that other scripts do not crash when looking to import the function.
    raise ValueError("Deprecated")
    return
def feature_selection_ks(df, num_features, return_similarity=False, key='label'):
    """
    #Feed:
    df with features AND ID and key column. Returns sorted Df of KS-similarities (first have lowest KS-similarity == best discriminators)
    Rank features based on Kolmogorov–Smirnov similarity, and take features that have the least interclass similarity
    Args:
        df: feature dataframe
        num_features: number of features to keep

    Returns: filtered feature dataframe
    if return_similarity is True, it will add a row with ID "Similarity"
    key specifiies which column is used for differentiatiion. Usually this will be label, but for subset experiments this might be 'easy', i.e. wether or not a sample belongs to the easy subset.

    """
    IDs = list(df['ID'])
    classes = list(df[key])
    df = df.drop(columns=['ID'])
    df = df.drop(columns=[key])
    df=df.reset_index(drop=True)
    features = df.to_numpy()
    ps = get_ks_similarity_interclass(features,classes)
    similarity_index = len(df.index)
    df.loc[similarity_index] = [*ps] #At last index we add importances.
    sorting_row = df.iloc[similarity_index]

    sorted_features = sorting_row.sort_values(ascending=True).index

    df = df[sorted_features[:num_features]]
    # for name in sorted_features[:num_features]:
    #     if name in list(df.columns):
    #         print(df[name][similarity_index])
    #     else:
    #         raise ValueError("QUE")

    if return_similarity:
        classes.append(0)
        IDs.append("Similarity")
    else:
        df=df.drop(similarity_index)
    #En haal deze 2 lines weer weg:


    df.insert(0, key, classes)
    df.insert(0, 'ID', IDs)
    df=df.copy()
    return df

def feature_selection_wd(df, num_features):
    """
    Rank features based on Wasserstein similarity, and take features that have the least interclass similarity
    Args:
        df: feature dataframe
        num_features: number of features to keep

    Returns: filtered feature dataframe

    """
    IDs = list(df['ID'])
    classes = df['label'].to_numpy()
    df = df.drop(columns=['ID'])
    df = df.drop(columns=['label'])
    df = df.reset_index(drop=True)
    features = df.to_numpy()
    ps = get_wd_similarity_interclass(features, classes)
    similarity_index = len(df.index)
    df.loc[similarity_index] = [*ps]  # At last index we add importances.
    sorting_row = df.iloc[similarity_index]
    sorted_features = sorting_row.sort_values().index
    df = df[sorted_features[:num_features]]
    df = df.drop(similarity_index)
    df.insert(0, 'label', classes)
    df.insert(0, 'ID', IDs)
    df = df.copy()
    return df


def distribution_similarity_WD(dist1, dist2):
    """
    Calculate the wasserstein similarity between 2 distributions
    Args:
        dist1: numpy array with values that make up distribution 1
        dist2: numpy array with values that make up distribution 2

    Returns: wasserstein similarity

    """
    dist1_normalized = dist1  # / np.sum(dist1)
    dist2_normalized = dist2  # / np.sum(dist2)
    emd = wasserstein_distance(dist1_normalized, dist2_normalized)
    similarity = 1 / (1 + emd)  # Normalize to the range [0, 1]
    return similarity
def get_wd_similarity_interclass(all_features, _all_classes):
    """Calculates interclass ws_similarity for all features"""
    ps2 = []
    for i in range(all_features.shape[-1]):
        feature = all_features[:, i]
        mal = feature[np.where(_all_classes == 1)]
        ben = feature[np.where(_all_classes == 0)]
        p = distribution_similarity_WD(mal, ben)
        ps2.append(p)
    return ps2



def KS_alpha(df,mod, num_features, alpha, return_similarity=False, key="label"):
    #alpha = np.clip(num_features,0,1)
    func = distribution_similarity_KS
    df['train'] = df['ID'].isin(mod.train_samples).astype(int) #Add column that specifies whether sample is in train or val
    exclude_cols = ['label', 'ID', 'train']
    sum_intraclass = []
    b_intraclass=[]
    m_intraclass=[]
    interclass = []
    # Loop through each column in the DataFrame
    df.loc['interclass'] = None
    df.loc['sum_intraclass'] = None

    for col in df.columns:
        if col not in exclude_cols:
            # Calculate the malignant intraclass value
            train_features = df[(df['label'] == 1) & (df['train'] == 1)][col]
            val_features = df[(df['label'] == 1) & (df['train'] == 0)][col]
            malignant_intraclass = func(train_features, val_features)
            m_intraclass.append(malignant_intraclass)

            # Calculate the benign intraclass value
            train_features = df[(df['label'] == 0) & (df['train'] == 1)][col]
            val_features = df[(df['label'] == 0) & (df['train'] == 0)][col]
            benign_intraclass = func(train_features, val_features)
            b_intraclass.append(benign_intraclass)

            # Calculate the sum of intraclass values
            sum_intraclass_value = ((1-malignant_intraclass) + (1-benign_intraclass)) / 2
            sum_intraclass.append(sum_intraclass_value)

            # Calculate the interclass value
            benign_features = df[(df['label'] == 0)][col]
            malignant_features = df[(df['label'] == 1)][col]
            interclass_value = func(benign_features, malignant_features)
            interclass.append(interclass_value)

            # Add the values to the appropriate rows in the DataFrame
            df.at['interclass', col] = interclass_value
            df.at['sum_intraclass', col] = sum_intraclass_value


            #Calculate new similarity with alpha
            interclass_value = df.loc['interclass', col]
            sum_intraclass_value = df.loc['sum_intraclass', col]

            # Calculate ks_similarity only if neither interclass nor sum_intraclass is None
            if pd.notna(interclass_value) and pd.notna(sum_intraclass_value):
                ks_similarity_value = alpha * interclass_value + (1 - alpha) * sum_intraclass_value
            else:
                ks_similarity_value = None

            # Assign the calculated value to the ks_similarity row
            df.at['ks_similarity', col] = ks_similarity_value
    df = df.sort_values(by='ks_similarity', axis=1, ascending=True)
    df.at['ks_similarity','ID'] = "Similarity"  #For compatibility with older version of this function.
    df=df.drop(columns=['train'])



    if not return_similarity:
        # Delete the rows 'interclass', 'sum_intraclass', and 'ks_similarity'
        df = df.drop(['interclass', 'sum_intraclass', 'ks_similarity'], axis=0)
    else:
        df = df.drop(['interclass', 'sum_intraclass'], axis=0)

    #Now, we onnly return the first num_columns features
    # Identify columns to include by excluding those in exclude_cols
    included_columns = [col for col in df.columns if col not in exclude_cols]
    # Select the first num_features columns from the filtered list
    selected_columns = included_columns[:num_features]
    # Identify columns from exclude_cols that are still present in the DataFrame
    remaining_excluded_cols = [col for col in reversed(exclude_cols) if col in df.columns] #Aka ID, label, etc.
    # Combine the selected columns with the remaining excluded columns
    final_columns = remaining_excluded_cols+selected_columns

    # Create a DataFrame with the final set of columns
    df_final = df[final_columns]

    # Return the DataFrame with the selected and remaining excluded columns
    return df_final

def get_ks_similarity_interclass(all_features, _all_classes):
    """Calculates interclass ks_similarity for all features"""
    # https://stats.stackexchange.com/questions/354035/how-to-compare-the-data-distribution-of-2-datasets
    _all_classes = np.array(_all_classes)
    ps2 = []
    for i in range(all_features.shape[-1]):
        feature = all_features[:, i]
        mal = feature[np.where(_all_classes == 1)]
        ben = feature[np.where(_all_classes == 0)]
        p = distribution_similarity_KS(mal, ben)
        ps2.append(p)
    return ps2

def distribution_similarity_KS(dist1, dist2):
    """
        Calculate the Kolmogorov–Smirnov similarity between 2 distributions
        Args:
            dist1: numpy array with values that make up distribution 1
            dist2: numpy array with values that make up distribution 2

        Returns: Kolmogorov–Smirnov similarity

        """
    ks_statistic, _ = ks_2samp(dist1, dist2)
    similarity = 1 - ks_statistic #Normalize
    return similarity
def plot_KS(df,mod, plot=True):
    """Plots interclass and intraclass KS similarities in bar chart"""
    df_m = filter_df(df, mod.train_samples)
    train_features = df_m.drop(columns=['ID', 'label']).to_numpy()
    train_classes = df_m['label'].to_numpy()
    df_m = filter_df(df, mod.val_samples)
    val_features = df_m.drop(columns=['ID', 'label']).to_numpy()
    val_classes = df_m['label'].to_numpy()
    radiomicFeatures3D, feature_names, classes, IDs = df_to_array(df)
    def compare_intra_class_vs_inter_class(t_features, t_classes, v_features, v_classes, all_features, _all_classes, func):
        ps = []
        for i in range(t_features.shape[-1]):
            feature = t_features[:, i]
            mal = feature[np.where(t_classes == 1)]
            feature = v_features[:, i]
            mal_v = feature[np.where(v_classes == 1)]
            p = func(mal, mal_v)
            ps.append(p)
        ps2 = []
        for i in range(all_features.shape[-1]):
            feature = all_features[:, i]
            mal = feature[np.where(np.array(_all_classes) == 1)]
            ben = feature[np.where(np.array(_all_classes) == 0)]
            p = func(mal, ben)
            ps2.append(p)
        return ps, ps2
    ps, ps2 = compare_intra_class_vs_inter_class(train_features, train_classes, val_features, val_classes,
                                                 radiomicFeatures3D, classes, distribution_similarity_KS)

    if plot:
        bins = 15
        plt.figure(figsize=(6, 5))

        # Create a subplot with two rows (2x1)
        #plt.subplot(2, 1, 1)

        # Plot the first histogram (ps)
        plt.hist(ps, bins, alpha=0.5, label='Mal vs Mal (train vs. val)')
        plt.hist(ps2, bins, alpha=0.5, label='Ben vs mal (train)')
        plt.legend()
        plt.xlabel("Similarity")
        plt.ylabel("Occurrence")
        plt.grid()
        plt.title("Feature-wise KS-similarity on Ovarian dataset")

        # # Create a subplot with two rows (2x1), and select the second subplot
        # plt.subplot(2, 1, 2)
        # plt.hist(ps_lung, bins, alpha=0.5, label='Mal vs Mal (train vs. val)')
        # plt.hist(ps2_lung, bins, alpha=0.5, label='Ben vs mal (train)')
        # plt.legend()
        # plt.xlabel("Similarity")
        # plt.ylabel("Occurrence")
        # plt.grid()
        # plt.title("Feature-wise KS-similarity on LIDC-IDRI dataset")  # Set a title for the second subplot
        plt.tight_layout()  # Ensure proper spacing between subplots
        #plt.savefig(os.path.join(DIR, 'Similarity_compare.eps'), format='eps')
        plt.show()

        raise ValueError("Plot generated, now exiting...")
    return ps, ps2
def standardize_features(df):
    radiomicFeatures3D, feature_names, classes, IDs = df_to_array(df)
    for i in range(np.shape(radiomicFeatures3D)[-1]):
        x = radiomicFeatures3D[:, i]
        #     radiomicFeatures3D[:,i] =  (x-np.min(x))/(np.max(x)-np.min(x))
        radiomicFeatures3D[:, i] = (x - np.mean(x)) / np.std(x)  # Standardize

    df = array_to_df(radiomicFeatures3D, feature_names, classes, IDs)
    return df
def load_data(data_path,csv_path, mod, crosstest=False): #Use cross_test for cross test experiment, in which case the full module and df will be returned
    '''Opens radiomic features in data_path. Requires 3DRadiomicFeatures.json. csv_path is not actually used...'''
    #df = pd.read_csv(csv_path, index_col=False)
    file_path = os.path.join(data_path, "3DRadiomicfeatures.json")
    with open(file_path, 'r') as file:
        loaded_data = json.load(file)
    # In the section below, we split of the last 4 entries as by default they encode class and test set characteristics
    radiomicFeatures3D = np.array(loaded_data['arr'])
    feature_names = loaded_data['names'][:-4]
    IDs = loaded_data['ids']
    classes = radiomicFeatures3D[:, -4]
    is_internal_test_set_tiny = radiomicFeatures3D[:, -3]
    is_internal_testset = radiomicFeatures3D[:, -2]
    is_external_testset = radiomicFeatures3D[:, -1]
    radiomicFeatures3D = radiomicFeatures3D[:, :-4]
    original_df = array_to_df(radiomicFeatures3D, feature_names, classes, IDs)
    original_df_no_borderlines = original_df[original_df['label'] != 2]  # Remove borderlines
    original_df_no_borderlines_no_nans = original_df_no_borderlines.dropna(axis='columns')



    original_df_no_borderlines_no_nans = standardize_features(original_df_no_borderlines_no_nans)
    train_val_list = [item for sublist in [mod.train_samples, mod.val_samples] for item in
                      sublist]  # Merge train and val for feature selection
    if not crosstest:
        train_val_df = filter_df(original_df_no_borderlines_no_nans, train_val_list)
        test_df = filter_df(original_df_no_borderlines_no_nans, mod.test_samples)
        df = train_val_df
        check_disjoint(df, test_df)
        return df, test_df
    else:
        return original_df_no_borderlines_no_nans, train_val_list, mod.test_samples





if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

