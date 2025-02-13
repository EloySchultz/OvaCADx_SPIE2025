
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
import matplotlib.pyplot as plt
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


def main(args):


    seed=args.seed
    csv_path = os.path.join(args.data_path_ovarian, "preprocessed_data.csv")
    assert os.path.exists(args.models_folder)
    nf_t = []
    #for k in tqdm(range(25)): #Nested Cross val handeled by module from Cris
    k=0
    mod = get_dataset(k, csv_path, seed, remove_outliers=False)
    df, df_test = load_data(args.data_path_ovarian, csv_path, mod, crosstest=False)
    df = pd.concat([df, df_test], ignore_index=True)
    ps_ov,ps2_ov = plot_KS(df,mod, plot=False)


    csv_path = os.path.join(args.data_path_lung, "preprocessed_data.csv")
    assert os.path.exists(args.models_folder)
    nf_t = []
    #for k in tqdm(range(25)): #Nested Cross val handeled by module from Cris
    k=0
    mod = get_dataset(k, csv_path, seed, remove_outliers=False)
    df, df_test = load_data(args.data_path_lung, csv_path, mod, crosstest=False)
    df = pd.concat([df, df_test], ignore_index=True)
    ps_lung,ps2_lung = plot_KS(df,mod, plot=False)

    bins = 15
    # plt.figure(figsize=(12, 2.5)) #6x5
    fig, axs = plt.subplots(1, 2, figsize=(10, 2.5))
    ax = axs[0]
    # Create a subplot with two rows (2x1)
    #ax.subplot(1, 2, 1)

    # Plot the first histogram (ps)
    ax.hist(ps_ov, bins, alpha=0.5,  label='Within-class')
    ax.hist(ps2_ov, bins, alpha=0.5, label='Between-class')
    ax.legend()
    ax.set_xlabel("KS-Similarity")
    ax.set_ylabel("$N$")
    ax.grid()
    ax.text(-0.02,1, chr(65 + 0), transform=ax.transAxes, fontsize=14, weight='bold',
            verticalalignment='bottom', horizontalalignment='right', fontname='Nimbus Roman')
    ax.set_title("Ovarian dataset")
    #
    # # # Create a subplot with two rows (2x1), and select the second subplot
    ax=axs[1]
    ax.hist(ps_lung, bins, alpha=0.5, label='Within-class')
    ax.hist(ps2_lung, bins, alpha=0.5, label='Between-class')
    ax.legend()
    ax.set_xlabel("KS-Similarity")
    ax.set_ylabel("$N$")
    ax.grid()
    ax.set_title("LIDC-IDRI dataset")  # Set a title for the second subplot
    ax.text(-0.02, 1, chr(65 + 1), transform=ax.transAxes, fontsize=14, weight='bold',
            verticalalignment='bottom', horizontalalignment='right', fontname='Nimbus Roman')
    plt.tight_layout()  # Ensure proper spacing between subplots
    plt.savefig(os.path.join(args.models_folder, 'Similarity_compare.pdf'), format='pdf')
    plt.show()

    return

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)