experiment_folder = "/home/eloy/Documents/Graduation_project/Code/MIL/experiments/experiment_3DCNN_OvarianRotate"
data_dir = "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new_FAST"
my_type = "OVARY"
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
    "--name '3CCNN_ovarian_rotate'",
    "--Desc '3CCNN_ovarian_rotate'",
    "--rotate 1"
]
command = f"python /home/eloy/Documents/Graduation_project/Code/MIL/train.py {' '.join(args)}"
os.system(command) #Uncomment to enable training



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
os.system(command) #Uncomment to enable testing


#Analyis
file_path = os.path.join(experiment_folder,"resultsMILCNN"+".csv")
args = ['--output_csv "{}"'.format(file_path)+' --data_dir "{}"'.format(data_dir)]
command = f"python /home/eloy/Documents/Graduation_project/Code/ovacadx/scripts/Radiomics/Analysis/ens_no_logits2.py {' '.join(args)}"+ "> '{}'".format(os.path.join(experiment_folder,"resultsMILCNN.txt"))
os.system(command)

#
# #Compile
# file_path = os.path.join(experiment_folder,"resultsMILCNN"+".txt")
# args = ['--output_txt "{}"'.format(file_path)]
# command = f"python /home/eloy/Documents/Graduation_project/Code/ovacadx/scripts/Radiomics/Analysis/compile_results.py {' '.join(args)}"
# os.system(command)