#Deze doet zowel training voor LIDC als OT voor SPIE paper (2 in 1, des nogus hendug)

import sys
sys.path.insert(1, '..')
import train
import test
import os
from datetime import datetime
experiment_folders =  "Radiomics_ovarian", "Radiomics_Lung"
data_dirs = "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new/", "/home/eloy/Documents/Data/LUNG_DATA/NIFTI-Processed_new"
import sys


for experiment_folder,data_dir in zip(experiment_folders,data_dirs):
    t_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", t_string)

    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)
    #Train
    pipeline = ['SELECT_PERCENTILE','PS','BLACK_BOX']
    features = ['10','40','0']
    command = 'python ../train.py --seed 0 --data_path '+str(data_dir)+' --models_folder "{}"'.format(experiment_folder) + ' --pipeline "{}"'.format(','.join(pipeline)) + ' --features "{}"'.format(','.join(features))
    print(command)
    # os.system(command)


    #Test
    #os.system('python ../test.py --seed 0 --data_path '+str(data_dir)+' --models_folder "{}"'.format(experiment_folder))
    #Analysis
    for filename in ["resultsNN","resultsSVM","resultsLR","resultsRF"]:
        file_path = os.path.join(experiment_folder,filename+".csv")
        print("Ensemble results for: " + file_path)
        command = 'python /home/eloy/Documents/Graduation_project/Code/ovacadx/scripts/Radiomics/Analysis/ens_no_logits2.py --seed 0 --data_dir "{}"'.format(data_dir) + ' --output_csv "{}"'.format(
                file_path)+'> "{}"'.format(os.path.join(experiment_folder,filename+".txt"))
        print(command)
        #os.system(command)