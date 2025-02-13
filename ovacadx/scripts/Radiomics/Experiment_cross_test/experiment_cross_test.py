import sys
sys.path.insert(1, '..')
import os
from datetime import datetime
experiment_folder_with_outliers="Radiomics_with_outliers_32"
experiment_folder_without_outliers="Radiomics_without_outliers_32"


#Please ensure that the pipeline in train.py is set to the following:
#Pipeline
        # df = feature_selection_ps(df, percentile=10)
        # df, nf = feature_selection_black_box_rf(df,mod, auto_select_kp = "ACC", show_graph=True)
        # nf_t.append(nf)
        #

t_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", t_string)

if not os.path.exists(experiment_folder_with_outliers):
    os.mkdir(experiment_folder_with_outliers)
if not os.path.exists(experiment_folder_without_outliers):
    os.mkdir(experiment_folder_without_outliers)
# os.system('python ../train.py --seed 0 --data_path "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new/" --models_folder "{}"'.format(experiment_folder_with_outliers))
# os.system('python ../train.py --seed 0 --remove_outliers --data_path "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new/" --models_folder "{}"'.format(experiment_folder_without_outliers))
# os.system('python test_cross_test.py --seed 0 --data_path "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new/" --models_folder "{}"'.format(experiment_folder_with_outliers))
# os.system('python test_cross_test.py --seed 0 --remove_outliers --data_path "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new/" --models_folder "{}"'.format(experiment_folder_without_outliers))

pp_path = "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new/"
command = 'python analysis_cross_test.py --seed 0 --data_path_with_outliers "{}"'.format(experiment_folder_with_outliers) + ' --data_path_without_outliers "{}"'.format(experiment_folder_without_outliers) + ' --preprocessed_path "{}"'.format(pp_path)
print(command)
os.system(command)
# #
