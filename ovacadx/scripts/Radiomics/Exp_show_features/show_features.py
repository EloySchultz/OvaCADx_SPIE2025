
from tqdm import tqdm
import os
import sys
sys.path.insert(1, '..')
import pandas as pd
from classifiers import train_classifiers, check_disjoint, save_models, load_models
import argparse
from pathlib import Path
from feature_selection import feature_selection_anova, feature_selection_pca, feature_selection_black_box_rf,feature_selection_wd,feature_selection_ks,feature_selection_ps, load_data, plot_KS, feature_selection_select_percentile
from utils import get_dataset
import importlib.util
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def get_args_parser():
    parser = argparse.ArgumentParser(description='Vizualize radiomics')
    parser.add_argument('--data_path_ovarian', type=str,
                        default='/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new')  # default='/path/to/output/folder', help='Path to output folder')
    parser.add_argument('--data_path_lung', type=str,
                        default='/home/eloy/Documents/Data/LUNG_DATA/NIFTI-Processed_new')
    parser.add_argument('--models_folder',type = str, default="/home/eloy/Documents/Graduation_project/Code/ovacadx/scripts/Radiomics/Experiment_plot_ks_SPIE/Radiomics_with_outliers_plotKS/")
    parser.add_argument('--seed', type = int, default = 0)

    parser.set_defaults(remove_outliers=False) #By default remove outliers
    return parser


def remove_outliers(data):
    """
    Removes outliers using the IQR method.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define bounds for outlier removal
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR

    # Filter the data
    return data[(data >= lower_bound) & (data <= upper_bound)]

def main(args):


    seed=args.seed

    for pth in [args.data_path_ovarian, args.data_path_lung]:
        csv_path = os.path.join(pth, "preprocessed_data.csv")
        assert os.path.exists(args.models_folder)
        nf_t = []
        #for k in tqdm(range(25)): #Nested Cross val handeled by module from Cris
        k=0
        mod = get_dataset(k, csv_path, seed, remove_outliers=False)
        df, df_test = load_data(pth, csv_path, mod, crosstest=False)



        n=16
        # df = pd.concat([df, df_test], ignore_index=True)
        # ps_ov,ps2_ov = plot_KS(df,mod, plot=False)
        df_filtered = feature_selection_ks(df,1300, return_similarity=True)
        df_filtered.drop(columns=["ID"])
        fig, axes = plt.subplots(4, 2, figsize=(12, 12))
        fig.tight_layout(pad=4.0)
        ks_similarity = df_filtered.iloc[-1, 2:]  # Extract KS-similarity values
        df_filtered = df_filtered.iloc[:-1]  # Remove the last row
        feature_names_output = []
        for i in range(n):
            # Get the feature name (column name)
            feature_name = df_filtered.columns[i + 2]  # Skip 'label' column, which is the first one

            # Get the feature values
            feature_values = df_filtered[feature_name]

            # Separate the values by class
            mal = feature_values[df_filtered['label'] == 1.0]
            ben = feature_values[df_filtered['label'] == 0.0]

            # Remove outliers
            mal_no_outliers = mal#remove_outliers(mal)
            ben_no_outliers = ben#remove_outliers(ben)

            # Create a DataFrame for plotting
            df = pd.DataFrame({"values": mal_no_outliers, "malignant": np.ones(len(mal_no_outliers))})
            df = pd.concat([df, pd.DataFrame({"values": ben_no_outliers, "malignant": np.zeros(len(ben_no_outliers))})])

            # Create a subplot with 2 rows and 4 columns (or any other layout based on n)
            plt.subplot(int(np.ceil(n / 4)), 4, i + 1)
            ax = sns.violinplot(data=df, x="malignant", y="values", cut=2)

            # Set the KS-similarity as the title
            ks_value = ks_similarity.iloc[i]
            ax.set_title(f"KS Similarity: {ks_value:.4f}")

            # Add bold alphabetic letter in the top right corner
            ax.text(-0.10, 1.13, chr(65 + i), transform=ax.transAxes, fontsize=20, fontweight='bold',
                    ha='left', va='top', family='Times New Roman')

            ax.grid()
            feature_names_output.append(f"{chr(65 + i)}: {feature_name}")
        plt.tight_layout()
        plt.show()

        print(", ".join(feature_names_output))

    #
    # csv_path = os.path.join(args.data_path_lung, "preprocessed_data.csv")
    # assert os.path.exists(args.models_folder)
    # nf_t = []
    # #for k in tqdm(range(25)): #Nested Cross val handeled by module from Cris
    # k=0
    # mod = get_dataset(k, csv_path, seed, remove_outliers=False)
    # df, df_test = load_data(args.data_path_lung, csv_path, mod, crosstest=False)
    # df = pd.concat([df, df_test], ignore_index=True)
    # ps_lung,ps2_lung = plot_KS(df,mod, plot=False)
    #
    # # Define vertical option
    # vertical = True  # Set to True for vertical arrangement, False for horizontal
    #
    # bins = 15
    #
    # # Adjust the layout based on the `vertical` flag
    # if vertical:
    #     fig, axs = plt.subplots(2, 1, figsize=(5.0, 4.5))  # Two rows, one column
    # else:
    #     fig, axs = plt.subplots(1, 2, figsize=(10, 2.5))  # One row, two columns
    #
    # ax = axs[0]
    # # Plot the first histogram (ps) for the Ovarian dataset
    # ax.hist(ps_ov, bins, alpha=0.5, label='Within-class')
    # ax.hist(ps2_ov, bins, alpha=0.5, label='Between-class')
    # ax.legend()
    # ax.set_xlabel("KS-Similarity")
    # ax.set_ylabel("$N$")
    # ax.grid()
    # ax.text(-0.02, 1, chr(65 + 0), transform=ax.transAxes, fontsize=14, weight='bold',
    #         verticalalignment='bottom', horizontalalignment='right', fontname='Times New Roman')
    # ax.set_title("Ovarian dataset")
    #
    # # Plot the second histogram (ps_lung) for the LIDC-IDRI dataset
    # ax = axs[1]
    # ax.hist(ps_lung, bins, alpha=0.5, label='Within-class')
    # ax.hist(ps2_lung, bins, alpha=0.5, label='Between-class')
    # ax.legend()
    # ax.set_xlabel("KS-Similarity")
    # ax.set_ylabel("$N$")
    # ax.grid()
    # ax.set_title("LIDC-IDRI dataset")
    # ax.text(-0.02, 1, chr(65 + 1), transform=ax.transAxes, fontsize=14, weight='bold',
    #         verticalalignment='bottom', horizontalalignment='right', fontname='Times New Roman')
    #
    # plt.tight_layout()  # Ensure proper spacing between subplots
    # plt.savefig(os.path.join(args.models_folder, 'Similarity_compare.pdf'), format='pdf')
    # print(args.models_folder)
    # plt.show()

    return

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

def show_distributions(m_features, my_classes, func):
    ps = []
    ks = []
    for i in range(107):
        feature = m_features[:, i]
        mal = feature[np.where(my_classes == 1)]
        ben = feature[np.where(my_classes == 0)]
        p = func(mal, ben)
        #     k,p=ks_2samp(mal, ben) #extract p-value
        ps.append(p)
    #     ks.append(k)

    plt.hist(ps, bins=15)
    plt.xlabel("Similarity")
    plt.ylabel("Occurrence")
    plt.show()

    ids = np.argsort(ps)
    n = 16
    # Create a 4x2 subplot grid
    fig, axes = plt.subplots(4, 2, figsize=(18, int(5 * (n / 4))))
    fig.tight_layout(pad=4.0)

    for i in range(n):
        ind = ids[i]  # Assuming ids contains the indices of the selected features
        feature_name = feature_names2[ind]
        similarity_value = ps[ind]  # Assuming ps contains the similarity values

        # print(f"Feature Name: {feature_name}, Similarity Value: {similarity_value}")

        feature = radiomicFeatures3D[:, ind]
        mal = feature[np.where(my_classes == 1)]
        ben = feature[np.where(my_classes == 0)]

        df = pd.DataFrame({"values": mal, "malignant": np.ones(len(mal))})
        df = pd.concat([df, pd.DataFrame({"values": ben, "malignant": np.zeros(len(ben))})])

        # Create a subplot with 2 rows and 4 columns
        plt.subplot(int(n / 4), 4, i + 1)
        ax = sns.violinplot(data=df, x="malignant", y="values")
        plt.title(f"Similarity: {similarity_value}\n{feature_name}")
        plt.grid()
    plt.tight_layout()
    plt.savefig("/home/eloy/Documents/Graduation_project/Figures/OVARY_radiomics.eps")
    plt.savefig("/home/eloy/Documents/Graduation_project/Figures/OVARY_radiomics.png")