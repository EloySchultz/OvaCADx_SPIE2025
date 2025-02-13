import sys
sys.path.insert(1, '..')
from tqdm import tqdm
import os
import pandas as pd
from classifiers import train_classifiers, check_disjoint, save_models, load_models, test_classifiers
from feature_selection import df_to_array,load_data, array_to_df
import argparse
from utils import get_dataset, filter_df
from pathlib import Path



#Samples that are outliers must always be in the test set for this cross analyiss.
#Je moet een csv maken voor test met outliers en test zonder outliers.
#De module moet altijd gelijk gecalled worden aan de training
#GEEN OUTLIERS --> GEEN OUTLIERS simpelweg 2 x dezelfde module callen zonder outliers
#GEEN OUTLIERS --> MET OUTLIERS Wederom 2x dezelfde module callen, maar nu de outliers toevoegen in de external test set
#GEEN OUTLIERS --> ALLEEN OUTLIERS, Zelfde als hierboven

#MET OUTLIERS --> GEEN OUTLIERS module callen zoals bij training, maar voor je de AUC berekent de outliers eruit halen
#MET OUTLIERS --> MET OUTLIERS module callen bij training
#MET OUTLIERS --> Alleen outliers = zelfde als hierboven, alleen de outliers meerekenen.


def get_args_parser():
    parser = argparse.ArgumentParser(description='Test classifiers')
    parser.add_argument('--data_path', type=str,
                        default='/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/')  # default='/path/to/output/folder', help='Path to output folder')
    parser.add_argument('--models_folder',type = str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--remove_outliers',  action='store_true') #MAKE SURE THAT THIS CORRESPONDS TO THE SETTING USED DURING TRAINING!
    parser.set_defaults(remove_outliers=False) #By default remove outliers
    return parser
def main(args):
    seed=args.seed
    remove_outliers = args.remove_outliers
    csv_path = os.path.join(args.data_path, "preprocessed_data.csv")
    all_results=[]
    assert args.models_folder != ""
    assert os.path.exists(args.models_folder)
    num_models = 4
    output_dfs = {
        'SVM': pd.DataFrame(),
        'RF': pd.DataFrame(),
        'LR': pd.DataFrame(),
        'NN': pd.DataFrame()
    }
    print("Running inference on TEST set and outputting logits in csv file")

    for outliers_in_testset in [False,True]:

        for k in tqdm(range(25)): #Nested Cross val handeled by module from Cris
            mod = get_dataset(k, csv_path, seed, remove_outliers) #always have remove outliers False, as they are filtered out manually below
            original_df_no_borderlines_no_nans, train_list, test_list = load_data(args.data_path, csv_path, mod,crosstest=True)
            o_df = pd.read_csv(csv_path)
            o_ids = o_df[o_df['outlier']==1]['tumor_id']


            #Move outliers into test set if they are to be excluded.
            if remove_outliers==True and outliers_in_testset == True:
                #Outliers must be added to test set
                test_list.extend(list(o_ids))
                assert len(test_list) == len(set(test_list))
            if remove_outliers==False and outliers_in_testset==False:
                #Outliers must be removed from test set
                test_list = set(test_list) - set(o_ids)

            df = filter_df(original_df_no_borderlines_no_nans, train_list)
            test_df = filter_df(original_df_no_borderlines_no_nans, test_list)
            check_disjoint(df,test_df)


            # Filter to only keep selected features. Note that in case of PCA, columns of both test and trainval will be replaced with PCx, hence no filtering is requireed.


            #Inference on test set
            model_dir = os.path.join(args.models_folder, "fold_"+str(k))
            models,pre_pca_feature_names,PCA_transform = load_models(model_dir)

            #Apply only features that have been included
            if list(test_df.columns.sort_values()) != list(pre_pca_feature_names):
                test_df = test_df[pre_pca_feature_names]

            #Applies feature selection.
            if PCA_transform != None:
                pca= PCA_transform
                num_features = PCA_transform.components_.shape[0]
                radiomicFeatures3D, feature_names, classes, IDs = df_to_array(test_df)
                X_test_pca = pca.transform(radiomicFeatures3D)
                PC_names = [f'PC{i + 1}' for i in range(num_features)]
                test_df = array_to_df(X_test_pca, PC_names, classes, IDs)


            logits_dfs = test_classifiers(models,test_df,k)
            for classifier in output_dfs.keys():
                if not output_dfs[classifier].empty:
                    output_dfs[classifier] = pd.merge(output_dfs[classifier], logits_dfs[classifier], on='tumor_id', how='outer')
                else:
                    output_dfs[classifier] = logits_dfs[classifier]

            # Display the results
        for classifier, df in output_dfs.items():
            df = output_dfs[classifier]
            # df_avl = df[df['tumor_id'].str.startswith('AVL')]
            # df_non_avl = df[~df['tumor_id'].str.startswith('AVL')]
            # df_non_avl.to_csv(os.path.join(args.models_folder,"results_"+classifier+".csv"),index=False)
            # df_avl.to_csv(os.path.join(args.models_folder, "results_EXT_" + classifier + ".csv"), index=False)
            if outliers_in_testset:
                str_ext = "WITH_OUTLIERS_"
            else:
                str_ext=""
            df.to_csv(os.path.join(args.models_folder, "results_CS_" +str_ext+ classifier + ".csv"), index=False)
        print("Test output csv files saved in: " + str(args.models_folder))



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)