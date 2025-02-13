
from tqdm import tqdm
import os
import pandas as pd
from classifiers import train_classifiers, check_disjoint, save_models, load_models
import argparse
from pathlib import Path
from feature_selection import feature_selection_anova, feature_selection_pca, feature_selection_black_box_rf,feature_selection_wd,feature_selection_ks,feature_selection_ps, load_data, plot_KS, feature_selection_select_percentile, KS_alpha
from utils import get_dataset
import matplotlib.pyplot as plt
import importlib.util
import sys
import os
def get_args_parser():
    parser = argparse.ArgumentParser(description='Vizualize radiomics')
    parser.add_argument('--data_path', type=str,
                        default='/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/')  # default='/path/to/output/folder', help='Path to output folder')
    parser.add_argument('--models_folder',type = str, default=None)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--remove_outliers',  action='store_true')
    parser.add_argument('--pipeline', type=str, required=True, help='Comma-separated list of feature selectors')
    parser.add_argument('--features', type=str, required=True,
                        help='Comma-separated list of number of features to keep for each selector')
    parser.add_argument('--alpha', type=float, help = 'alpha to use for ks_alpha feature selector. 1=interclass, 0 = intraclass. You can mix them.')
    parser.set_defaults(remove_outliers=False) #By default remove outliers
    return parser




def main(args):
    seed=args.seed
    remove_outliers = args.remove_outliers
    csv_path = os.path.join(args.data_path, "preprocessed_data.csv")

    all_results=[]
    assert os.path.exists(args.models_folder)
    nf_t = []
    for k in tqdm(range(25)): #Nested Cross val handeled by module from Cris
        mod = get_dataset(k, csv_path, seed, remove_outliers)
        df, _ = load_data(args.data_path, csv_path, mod)
        # print(remove_outliers)
        if k==0:
            print("TrainVal Length:" + str(len(df)))
        PCA_transform = None

        #Pipeline
        pipeline_steps = args.pipeline.split(',')
        num_features = list(map(int, args.features.split(',')))
        # Check if lengths match
        if len(pipeline_steps) != len(num_features):
            raise ValueError("The number of feature selectors must match the number of feature counts.")
        pre_pca_feature_names = []
        for step, num,i in zip(pipeline_steps, num_features, range(len(num_features))):
            if step == 'BLACK_BOX':
                df, _ = feature_selection_black_box_rf(df, mod, auto_select_kp="ACC", show_graph=True)
            elif step == 'PCA':
                pre_pca_feature_names = df.columns
                df, PCA_transform  = feature_selection_pca(df, num_features=num)
            elif step=="SELECT_PERCENTILE":
                df = feature_selection_select_percentile(df,percentile=num)
            elif step == 'PS':
                df = feature_selection_ps(df, num_features=num)
            elif step == 'ANOVA':
                df = feature_selection_anova(df, num_features=num)
            elif step == 'KS':
                df = feature_selection_ks(df, num_features=num)
            elif step == 'WD':
                df = feature_selection_wd(df, num_features=num)
            elif step == 'KS_alpha':
                df = KS_alpha(df,mod,num_features=num,alpha=args.alpha)  #here num[0] is number of features, num[1] is alpha
            elif step == "plot_KS":
                df = plot_KS(df,mod)

            else:
                raise ValueError(f"Unknown feature selection step: {step}")
        if len(pre_pca_feature_names)==0:
            pre_pca_feature_names = df.columns #If no pca was performed, simply save the columns

        #Train classifiers and save models
        results,models = train_classifiers(df,mod, plot=False, verbose=False)
        all_results.append(results)
        model_dir = os.path.join(args.models_folder, "fold_"+str(k))
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        save_models(models,model_dir,pre_pca_feature_names,PCA_transform)

    combined_val_results = pd.concat(all_results)
    print("Val results:", combined_val_results.groupby("Classifier").mean())
    return combined_val_results.groupby("Classifier").mean(), nf_t

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)