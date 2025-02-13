experiment_folder = "/home/eloy/Documents/Graduation_project/Code/MIL/experiments/experiment_ov_rotate"
data_dir = "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new_FAST"
my_type="OVARY"
import os
args = [
    "--data_dir /home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new_FAST",
    "--results_dir " +experiment_folder,
    "--num_workers 15",
    "--cross_validate True",
    "--k_start 0",
    "--k_end 25",
    "--lr 0.0000075",
    "--name '100Epoch_rotate'",
    "--Desc 'Ovarian 100 epoch with rotation'",
    "--rotate 1"
]

# Construct the command string
command = f"python /home/eloy/Documents/Graduation_project/Code/MIL/train.py {' '.join(args)}"

# Execute the command
# os.system(command) #Uncomment to enable training




#Testing
args = [
"--type " + my_type,
"--data_dir "+data_dir,
"--results_dir " + str(experiment_folder),
"--seed 0",
"--checkpoint_substring 'best_auc'"
]
command = f"python /home/eloy/Documents/Graduation_project/Code/MIL/test.py {' '.join(args)}"
os.system(command)


#Analyis
file_path = os.path.join(experiment_folder,"resultsMILCNN"+".csv")
args = ['--output_csv "{}"'.format(file_path)+' --data_dir "{}"'.format(data_dir)]
command = f"python /home/eloy/Documents/Graduation_project/Code/ovacadx/scripts/Radiomics/Analysis/ens_no_logits2.py {' '.join(args)}"+ "> '{}'".format(os.path.join(experiment_folder,"resultsMILCNN.txt"))
os.system(command)
