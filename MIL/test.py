import os
#https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy, https://github.com/pytorch/pytorch/issues/15808
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
#proposed deadlock fix in https://github.com/pytorch/pytorch/issues/1355
import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
from tqdm import tqdm
import re
#import monai.data as data
import monai.transforms as transforms
import pandas as pd
import sys
import argparse
from datetime import datetime
import torch.optim as optim
from model import Attention, GatedAttention, ResNET3D
from dataloader import create_dataloader

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from monai.utils import set_determinism
import random
import sys
sys.path.insert(1, '/home/eloy/Documents/Graduation_project/Code/ovacadx/') #Path to ovacadx module made by Cris
import scripts.Radiomics.utils as utils
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  #Needed for determinism (reproducibility, https://discuss.pytorch.org/t/random-seed-with-external-gpu/102260/18)

def get_args_parser():
    parser = argparse.ArgumentParser(description='MIL CNN test script')
    parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')
    parser.add_argument('--type', type=str, default='OVARY',
                        help='Specify which data type to train on. Lung or ovary. This is important for clipping settings of the HUs.')
    parser.add_argument('--data_dir', type=str, default='', help='Dataset directory containing preprocessed_data.csv')
    parser.add_argument('--results_dir', type=str, default="", help='Where results were saved. ')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for the dataloader.')
    parser.add_argument('--seed', type=int, default=0, help='Set seed for reproducibility')
    parser.add_argument('--k', type=int, default=0, help='Set seed for Kfold reproducibility')  # (script Cris)
    parser.add_argument('--Desc', type=str, default="", help='Description of your training')
    parser.add_argument('--checkpoint_substring', type=str, default="PUT SUBSTRING HERE", help='Substring for checkpoint name')
    return parser





def init_distributed_mode():
    dist_url = "env://"
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        rank, gpu, world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    torch.cuda.set_device(gpu)
    print('| distributed init (rank {}): {}'.format(
        rank, dist_url), flush=True)


def test(df, k, args,now,g):
    print("Evaluating fold " + str(k))
    checkpoint_substring = args.checkpoint_substring

    fold_output_dir = os.path.join(args.results_dir, "fold " + str(k))
    if not os.path.exists(fold_output_dir):
        raise ValueError("Folder "+str(fold_output_dir)+ " does not exist.")

    if args.type == "OVARY":
        clip_min = -100
        clip_max = 300
    elif args.type == "LUNG": #https://www.researchgate.net/publication/369865428_CT_Lung_Nodule_Segmentation_A_Comparative_Study_of_Data_Preprocessing_and_Deep_Learning_Models#pf2)
        clip_min = -1000
        clip_max = 400
    else:
        raise ValueError("Unkown image type, so cannot set clipping settings.")
    #Transforms
    test_transform = transforms.Compose([
        transforms.LoadImaged(keys=['image', 'segmentation']),
        transforms.EnsureChannelFirstd(keys=['image', 'segmentation'], channel_dim="no_channel"),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=clip_min, a_max=clip_max, b_min=0, b_max=1, clip=True),
        transforms.MaskIntensityd(keys=['image'], mask_key='segmentation'),
        transforms.CropForegroundd(keys=['image'], source_key='image', allow_smaller=True),
        # 1000,1000,0 voor alleen z crop margin=(1000,1000,0)
        transforms.SpatialPadd(keys=['image'], spatial_size=(512, 512, -1)),
        transforms.Resized(keys=['image'], spatial_size=(400, 400, -1)),
        # Slightly reduce size due to memory constraints
        transforms.ToTensord(keys=['image'])
    ])

    test_loader = create_dataloader(df, test_transform,False,args,g)




    #Find correct model checkpoint to load:
    files = os.listdir(fold_output_dir)
    # Filter the files containing the substring
    matching_files = [file for file in files if checkpoint_substring in file]
    if len(matching_files) == 0:
        raise ValueError("No file found that contains "+str(checkpoint_substring))
    if len(matching_files)>1:
        print("Warning, selecting first file containing " + str(checkpoint_substring))
    model_to_load = os.path.join(fold_output_dir,matching_files[0])

    checkpoint = torch.load(model_to_load, map_location="cpu")
    #load args
    namespace_dict = vars(checkpoint['args'])
    parser.set_defaults(**namespace_dict)
    checkpoint_args = parser.parse_args([])
    MIL=True
    if checkpoint_args.model == 'attention':
        model = Attention()
    elif checkpoint_args.model == 'gated_attention':
        model = GatedAttention()
    elif checkpoint_args.model == "3DCNN":
        model = ResNET3D(torch.nn.MSELoss())
        MIL = False
    else:
        raise ValueError('Unkown model')
    model.load_state_dict(checkpoint["model"])
    if True:
        model.cuda()
    #Test loop

    logits_dfs = {
        'MILCNN': pd.DataFrame()}
    if MIL:
        model.eval()  #Disables batch normalization
    prob = {'True': [], 'Pred': []}
    num_batches = len(test_loader)
    test_loss=0
    progress_bar = tqdm(enumerate(test_loader), total=num_batches)
    with torch.no_grad():
        ID_col = []
        for batch_idx, my_data in progress_bar:
            if MIL:
                img=my_data['image'].permute(0,4,1,2,3)
            else:
                img = my_data['image']
            ID_col.append(my_data['tumor_id'][0])
            img=img.cuda()
            bag_label = my_data['label'].cuda()

            loss, _, y_prob = model.calculate_objective(img, bag_label)
            prob['True'].append(bag_label.numpy(force=True))
            probability = np.squeeze(y_prob.numpy(force=True))
            prob['Pred'].append(probability)
            test_loss += loss  # .data[0]
            torch.cuda.synchronize()

        logits_dfs['MILCNN'] = pd.DataFrame({
            'tumor_id': ID_col,
            str(k): prob['Pred']
        })
        return logits_dfs

def get_fold_numbers(directory):
    # List all contents of the directory
    contents = os.listdir(directory)

    # Regular expression to match 'fold' followed by a number
    pattern = re.compile(r'fold (\d+)')

    # List to store the numbers
    fold_numbers = []

    # Iterate through the contents of the directory
    for item in contents:
        match = pattern.match(item)
        if match:
            # Extract the number and convert it to an integer
            fold_numbers.append(int(match.group(1)))

    return np.sort(fold_numbers)


def main(args):

    #https: // stackoverflow.com / questions / 63221468 / runtimeerror - dataloader - worker - pid - 27351 - is -killed - by - signal - killed
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    # Training settings
    now = str(datetime.now())
    init_distributed_mode()

    #From: https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(args.seed, use_deterministic_algorithms=True) #MONAI


    #https://pytorch.org/docs/stable/notes/randomness.html
    g = torch.Generator() #Needed for reproduciblility
    g.manual_seed(args.seed)

    #Find directory folders
    if not os.path.exists(args.results_dir):
        raise ValueError("Directory results_dir ("+str(args.results_dir)+") (where to save experiment) does not yet exist. Please create a valid directory")


    k_numbers = [int(k) for k in get_fold_numbers(args.results_dir)]
    output_dfs = {
        'MILCNN': pd.DataFrame()
    }
    output_dfs_val = {
        'MILCNN': pd.DataFrame()
    }
    for k in k_numbers:
        csv_file = os.path.join(args.data_dir, "preprocessed_data.csv")
        seed = args.seed

        df = pd.read_csv(csv_file)
        df = df.loc[df['label'] < 2]  # Fiter out borderlines
        mod = utils.get_dataset(k, csv_file, seed, remove_outliers=False)  # Never remove outliers
        df_val = df.loc[df['tumor_id'].isin(mod.val_samples)]
        df_train = df.loc[df['tumor_id'].isin(mod.train_samples)]
        df_test = df.loc[df['tumor_id'].isin(mod.test_samples)]

        logits_dfs= test(df_test,k,args,now,g)
        if not output_dfs['MILCNN'].empty:
            output_dfs['MILCNN'] = pd.merge(output_dfs['MILCNN'], logits_dfs['MILCNN'], on='tumor_id',
                                              how='outer')
        else:
            output_dfs['MILCNN'] = logits_dfs['MILCNN']

        logits_dfs = test(df_val, k, args, now, g)
        if not output_dfs_val['MILCNN'].empty:
            output_dfs_val['MILCNN'] = pd.merge(output_dfs_val['MILCNN'], logits_dfs['MILCNN'], on='tumor_id',
                                            how='outer')
        else:
            output_dfs_val['MILCNN'] = logits_dfs['MILCNN']
    output_dfs['MILCNN'].to_csv(os.path.join(args.results_dir,"resultsMILCNN.csv"),index=False)
    output_dfs_val['MILCNN'].to_csv(os.path.join(args.results_dir, "VAL_resultsMILCNN.csv"), index=False)
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)



# https://github.com/pytorch/pytorch/issues/1355#issuecomment-907986126 ??