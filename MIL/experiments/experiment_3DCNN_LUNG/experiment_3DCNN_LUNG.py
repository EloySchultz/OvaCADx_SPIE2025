experiment_folder = "/home/eloy/Documents/Graduation_project/Code/MIL/experiments/experiment_3DCNN_LUNG"
data_dir = "/home/eloy/Documents/Data/LUNG_DATA/NIFTI-Processed_new_FAST/"
my_type = "LUNG"
import os
#Training
args = [
    "--type " + my_type,
    "--data_dir "+data_dir,
    "--results_dir " +experiment_folder,
    "--num_workers 15",
    "--model '3DCNN'",
    "--cross_validate True",
    "--seed 0",
    "--k_start 0",
    "--k_end 25",
    "--lr 0.0000750",
    "--name '3CCNN_lung_baseline'",
    "--Desc '3CCNN_lung_baseline'"
]
command = f"python /home/eloy/Documents/Graduation_project/Code/MIL/train.py {' '.join(args)}"
# os.system(command)  #Uncomment to enable training



#Testing
args = [
"--type " + my_type,
"--data_dir "+data_dir,
"--results_dir " + str(experiment_folder),
"--seed 0",
"--checkpoint_substring 'best_auc'"
]
command = f"python /home/eloy/Documents/Graduation_project/Code/MIL/test.py {' '.join(args)}"
print(args)
# os.system(command) #Uncomment to enable testing


#Analyis
file_path = os.path.join(experiment_folder,"resultsMILCNN"+".csv")
args = ['--output_csv "{}"'.format(file_path)+' --data_dir "{}"'.format(data_dir)]
command = f"python /home/eloy/Documents/Graduation_project/Code/ovacadx/scripts/Radiomics/Analysis/ens_no_logits2.py {' '.join(args)}"+ "> '{}'".format(os.path.join(experiment_folder,"resultsMILCNN.txt"))
print(command)
os.system(command)


# #Compile
# file_path = os.path.join(experiment_folder,"resultsMILCNN"+".txt")
# args = ['--output_txt "{}"'.format(file_path)]
# command = f"python /home/eloy/Documents/Graduation_project/Code/ovacadx/scripts/Radiomics/Analysis/compile_results.py {' '.join(args)}"
# os.system(command)