import os
#https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy, https://github.com/pytorch/pytorch/issues/15808
# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
#proposed deadlock fix in https://github.com/pytorch/pytorch/issues/1355
from dataloader import create_dataloader
import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
from tqdm import tqdm
import monai.data as data
import monai.transforms as transforms
import pandas as pd
import sys
import argparse
from datetime import datetime
import torch.optim as optim
from model import Attention, GatedAttention, ResNET3D
import wandb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from monai.utils import set_determinism
import random
import sys
from helpers import show
sys.path.insert(1, '/home/eloy/Documents/Graduation_project/Code/ovacadx/')
import scripts.Radiomics.utils as utils
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  #Needed for determinism (reproducibility, https://discuss.pytorch.org/t/random-seed-with-external-gpu/102260/18)

def get_args_parser():
    parser = argparse.ArgumentParser(description='MIL CNN train script')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0000075, metavar='LR',
                        # This learning rate passes the overfit test: 0.00005 (21-03-2024). The 0.000033 was found using cos LR schedule (23-03-2024)
                        help='learning rate (default: 0.0005)')  # 0.0000005
    parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention or 3DCNN. 3DCNN will disable MIL')
    parser.add_argument('--name', type=str, default='Test',
                        help='Choose an informative name to help you recognize your training')
    parser.add_argument('--type', type=str, default='OVARY',
                        help='Specify which data type to train on. Lung or ovary. This is important for clipping settings of the HUs.')
    parser.add_argument('--data_dir', type=str, default='', help='Dataset directory containing preprocessed_data.csv')
    parser.add_argument('--results_dir', type=str, default="", help='Where to save the results')
    parser.add_argument('--cross_validate', type=bool, default=False,
                        help='Train all 25 train-val splits. Overrides --k')
    parser.add_argument('--k_start', type=int, default=0,
                        help='Set to start k at k_start, skipping earlier iterations (for cross validation)')
    parser.add_argument('--k_end', type=int, default=25,
                        help='Set to end k at k_end, skipping later iterations (for cross validation)')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for the dataloader.')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation occurs every N epochs')
    parser.add_argument('--checkpoint', type=str, default="", help='Path to load checkpoint from.')
    parser.add_argument('--seed', type=int, default=0, help='Set seed for reproducibility')
    parser.add_argument('--k', type=int, default=0, help='Set seed for Kfold reproducibility')  # (script Cris)
    parser.add_argument('--Desc', type=str, default="", help='Description of your training')
    parser.add_argument('--rotate', type=int, default=0, help='[EXPERIMENTAL]: Apply 3D rotation augmentation')  #
    return parser



def init_wandb(args,k,group="Default_group",name=""):
    hyperparameters = vars(args)
    hyperparameters['Current_fold'] = k
    if name=="":
        wandb.init(
            # set the wandb project where this run will be logged
            project="Graduation_project_CADX",
            group=group,
            # track hyperparameters and run metadata
            config=hyperparameters
        )
    else:
        wandb.init(
            # set the wandb project where this run will be logged
            project="Graduation_project_CADX",
            group=group,
            name=name,
            # track hyperparameters and run metadata
            config=hyperparameters
        )
    wandb.config.update(args)
    return

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
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    torch.cuda.set_device(gpu)
def train_with_split_dummy(k, args,now,g):
    for epoch in range(0, args.epochs + 1):
        wandb.log({"test": random.random(),"zest":random.random()})

def train_fold(k, args,now,g):
    csv_file = os.path.join(args.data_dir,"preprocessed_data.csv")
    seed = args.seed
    if not os.path.exists(args.results_dir):
        raise ValueError("Directory results_dir ("+str(args.results_dir)+") (where to save experiment) does not yet exist. Please create a valid directory")

    fold_output_dir = os.path.join(args.results_dir, "fold " + str(k))
    if not os.path.exists(fold_output_dir):
        os.makedirs(fold_output_dir)
    else:
        print(str(fold_output_dir) + " already exists.")
        # resp = input("Directory already exists. Type y to overwrite...")
        # if resp != "y":
        #     print("Aborting")
        #     sys.exit()
        print("Overwriting...")
    print("Logging to " + str(fold_output_dir))

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
    if args.rotate==0:
        train_transform = transforms.Compose([
            transforms.LoadImaged(keys=['image', 'segmentation']),
            transforms.EnsureTyped(keys="image"),
            transforms.EnsureChannelFirstd(keys=['image', 'segmentation'], channel_dim="no_channel"),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=clip_min, a_max=clip_max, b_min=0, b_max=1, clip=True),
            # -100 300 voor OVARY, -1000 400 voor LUNG (
            transforms.MaskIntensityd(keys=['image'], mask_key='segmentation'),
            transforms.CropForegroundd(keys=['image'], source_key='image', allow_smaller=True),
            # 1000,1000,0 voor alleen z crop
            transforms.RandFlipd(keys=['image'], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=['image'], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=['image'], prob=0.5, spatial_axis=2),
            # transforms.Rand3DElasticd(keys=['image'],sigma_range=[7,7], magnitude_range=[300,300],prob=1),
            # transforms.RandRotated(keys=['image'],range_x=np.pi/6, range_y=np.pi/6, range_z=np.pi/6, keep_size=False,mode="bilinear", padding_mode='zeros', prob=0.8),
            transforms.CropForegroundd(keys=['image'], source_key='image', allow_smaller=True),
            transforms.SpatialPadd(keys=['image'], spatial_size=(512, 512, -1)),
            transforms.Resized(keys=['image'], spatial_size=(400, 400, -1)),  # Slightly reduce size due to memory c
            transforms.ToTensord(keys=['image'])
        ])
    elif args.rotate==1:
        train_transform = transforms.Compose([
            transforms.LoadImaged(keys=['image', 'segmentation']),
            transforms.EnsureTyped(keys="image"),
            transforms.EnsureChannelFirstd(keys=['image', 'segmentation'], channel_dim="no_channel"),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=clip_min, a_max=clip_max, b_min=0, b_max=1, clip=True),
            # -100 300 voor OVARY, -1000 400 voor LUNG (
            transforms.MaskIntensityd(keys=['image'], mask_key='segmentation'),
            transforms.CropForegroundd(keys=['image'], source_key='image', allow_smaller=True),
            # # 1000,1000,0 voor alleen z crop
            transforms.RandFlipd(keys=['image'], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=['image'], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=['image'], prob=0.5, spatial_axis=2),
            # # transforms.Rand3DElasticd(keys=['image'],sigma_range=[7,7], magnitude_range=[300,300],prob=1),
            transforms.RandRotated(keys=['image'],range_x=np.pi/6, range_y=np.pi/6, range_z=np.pi/6, keep_size=False,mode="bilinear", padding_mode='zeros', prob=0.8),
            transforms.CropForegroundd(keys=['image'], source_key='image', allow_smaller=True),
            transforms.SpatialPadd(keys=['image'], spatial_size=(512, 512, -1)),
            transforms.Resized(keys=['image'], spatial_size=(400, 400, -1)),  # Slightly reduce size due to memory c
            transforms.ToTensord(keys=['image'])
        ])
    else:
        raise ValueError("Unknown value for --rotate")


    df = pd.read_csv(csv_file)
    df = df.loc[df['label']<2] #Fiter out borderlines

    mod = utils.get_dataset(k, csv_file, seed, remove_outliers = False) #Never remove outliers
    df_val = df.loc[df['tumor_id'].isin(mod.val_samples)]
    df_train = df.loc[df['tumor_id'].isin(mod.train_samples)]
    df_test = df.loc[df['tumor_id'].isin(mod.test_samples)]

    train_loader = create_dataloader(df_train, train_transform,True,args,g)
    val_loader = create_dataloader(df_val, test_transform,False,args,g)
    test_loader = create_dataloader(df_test, test_transform,False, args,g) #Not used here obviously.

    if args.model == 'attention':
        model = Attention()
        MIL=True
    elif args.model == 'gated_attention':
        model = GatedAttention()
        MIL = True
    elif args.model == "3DCNN":
        model = ResNET3D(torch.nn.MSELoss())
        MIL = False
    else:
        raise ValueError('Unkown model')
    if True:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    start_epoch=1
    best_auc = 0
    best_auc_acc = 0
    best_auc_val_loss = np.inf
    best_val_loss = np.inf
    if args.checkpoint != "":
        print("LOADING CHECKPOINT FROM: "+str(args.checkpoint))

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        cpt_path = args.checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        args = checkpoint['args']
        args.checkpoint = cpt_path
        best_auc = checkpoint['best_auc']
        best_auc_acc = checkpoint['best_auc_acc']
        best_auc_val_loss = checkpoint['best_auc_val_loss']
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch']
    #Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        if MIL:
            model.eval() #Disables batchnorm during training! (Needed due to variable batch size nature of MIL)
        train_loss = 0.
        prob = {'True': [], 'Pred': []}
        num_batches = len(train_loader)
        progress_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f'Epoch {epoch}')
        # a=7
        # b=7
        # c=300
        # d=300
        t=0
        for batch_idx, my_data in progress_bar:
            #For overfitting test (to tuno LR)
            # if t>20:
            #     break # This can be uncommented to only output the first 20 slices, which is useful in debugging the transformations below
            t+=1

            # # The commented code below can be used to manually debug the MONAI transformations and check to make sure that the images look good before going into the model
            # # If you uncomment the code below, make sure to REMOVE the transformation you are testing from the train_transform!!!
            # # clip_min=--1000
            # # clip_max = 400
            # clipper = transforms.ScaleIntensityRange(a_min=clip_min, a_max=clip_max, b_min=0, b_max=1, clip=True)
            # cropper = transforms.CropForeground()#transforms.Rand3DElastic(sigma_range=[a,b], magnitude_range=[c,d],prob=1)#
            # rotatorr = transforms.RandRotate(range_x=np.pi/6, range_y=np.pi/6, range_z=np.pi/6, keep_size=False,mode="bilinear", padding_mode='zeros', prob=1)  #scale_range=(0.9, 0.1) #Rotates in axial plane, scaling removed for now)
            # padder = transforms.SpatialPad(spatial_size=(512, 512, -1))
            # test_img =my_data['image'][0]
            # grid_mask = my_data['segmentation']
            #
            # if t<7:
            #     continue
            # #insert grid
            # width = test_img.shape[1]
            # height = test_img.shape[2]
            # depth = test_img.shape[3]
            # grid_spacing = 15
            # grid_mask = torch.ones((1, width, height, depth))
            # grid_mask[:, ::grid_spacing, :, :] = 0  # Vertical lines
            # grid_mask[:, :, ::grid_spacing, :] = 0  # Horizontal lines
            # # Apply the grid mask to the tensor
            # # test_img = test_img * grid_mask
            # test_img_org = test_img#.copy()
            # print(test_img.min(), test_img.max())
            # print(test_img.dtype)
            # test_img = clipper(test_img)
            # print(test_img.min(), test_img.max())
            # print(test_img.dtype)
            # test_img = cropper(test_img)
            # test_img = rotatorr(test_img)
            # test_img = cropper(test_img)
            # test_img = padder(test_img)
            #
            #
            # show(test_img.unsqueeze(0))

            if MIL:
                img=my_data['image'].permute(0,4,1,2,3)
            else:
                img = my_data['image']
            img=img.cuda()
            bag_label = my_data['label'].cuda()


            # reset gradients
            optimizer.zero_grad()
            # calculate loss and metrics
            loss, _, y_prob = model.calculate_objective(img, bag_label)
            prob['True'].append(bag_label.numpy(force=True))
            prob['Pred'].append(np.squeeze(y_prob.numpy(force=True)))
            train_loss += loss  # .data[0]

            # backward pass
            loss.backward()
            # step
            optimizer.step()
            torch.cuda.synchronize()


        # calculate loss and error for epoch
        train_loss /= num_batches
        train_auc = roc_auc_score(prob['True'], prob['Pred'])

        train_acc = 0
        for i in range(1000):
            train_acc = max(train_acc, accuracy_score(prob['True'], (np.array(prob['Pred']) > i / 1000) * 1)) #Just to get an estimate of the max accuracy

        print('\n Train, Loss: {:.4f}, Train AUC: {:.4f},  Train ACC: {:.4f}'.format(train_loss.numpy(force=True), train_auc,
                                                                     train_acc))
        #Validation loop
        if (epoch % args.val_freq ==0) or (epoch > args.epochs-1):
            val_loss = 0.
            prob = {'True': [], 'Pred': []}
            num_batches = len(val_loader)
            # Create a progress bar
            progress_bar = tqdm(enumerate(val_loader), total=num_batches, desc=f'VALIDATING Epoch {epoch}')
            t = 0
            with torch.no_grad():
                for batch_idx, my_data in progress_bar:
                    #For overfitting test (to tuno LR)
                    # if t>30:
                    #     break
                    # t+=1

                    if MIL:
                        img = my_data['image'].permute(0, 4, 1, 2, 3)
                    else:
                        img = my_data['image']
                    #print(img.shape)
                    img = img.cuda()
                    bag_label = my_data['label'].cuda()


                    # calculate loss and metrics
                    loss, _, y_prob = model.calculate_objective(img, bag_label)
                    prob['True'].append(bag_label.numpy(force=True))
                    prob['Pred'].append(np.squeeze(y_prob.numpy(force=True)))
                    val_loss += loss  # .data[0]
            # calculate loss and error for epoch
            val_loss /= num_batches
            val_auc = roc_auc_score(prob['True'], prob['Pred'])  # [(a,b) for a,b in zip(prob['True'],prob['Pred'])]

            malignant_indices= np.nonzero(np.array(prob['True']))[0]
            malignant_predictions = [prob['Pred'][i] for i in malignant_indices]
            benign_indices = np.where(np.array(prob['True']) == 0)[0]
            benign_predictions = [prob['Pred'][i] for i in benign_indices]
            val_acc = 0
            for i in range(100):
                val_acc = max(val_acc, accuracy_score(prob['True'], (np.array(prob['Pred']) > i / 100) * 1))

            print('\n Val, Loss: {:.4f}, Val AUC: {:.4f},  Val ACC: {:.4f}'.format(val_loss.numpy(force=True),
                                                                                         val_auc,val_acc))
            if val_auc > best_auc or (val_auc == best_auc and val_acc>best_auc_acc):
                best_auc=val_auc
                best_auc_acc = val_acc
                best_auc_val_loss = val_loss
                save_dict = {
                    "args": args,
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_auc_val_loss": best_auc_val_loss,
                    "best_auc_acc": best_auc_acc,
                    "best_auc": best_auc,
                    "best_val_loss": best_val_loss
                }
                torch.save(save_dict, os.path.join(fold_output_dir, args.name+ str("_checkpoint_best_auc.pth.tar")))

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                save_dict = {
                    "args": args,
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_auc_val_loss": best_auc_val_loss,
                    "best_auc_acc": best_auc_acc,
                    "best_auc": best_auc,
                    "best_val_loss": best_val_loss
                }
                torch.save(save_dict, os.path.join(fold_output_dir, args.name+ str("_checkpoint_best_val_loss.pth.tar")))

            print(f'Max AUC so far: {best_auc:.2f}')
            lr = scheduler.get_last_lr()[0]
            log_dict = {"Epoch": epoch,"Learning rate": lr,"Train acc": train_acc, "Train auc": train_auc, "Train loss": train_loss.numpy(force=True),
                       "Val acc": val_acc, "Val auc": val_auc, "Val loss": val_loss.numpy(force=True)}

            log_dict.update({f"Benign_{i}": benign_predictions[i] for i in range(min(10, len(benign_predictions)))})
            log_dict.update({f"Malignant_{i}": malignant_predictions[i] for i in range(min(10, len(malignant_predictions)))})
            wandb.log(log_dict)
        scheduler.step()
    #After training:

    save_dict = {
        "args": args,
        "epoch": epoch ,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_auc_val_loss": best_auc_val_loss,
        "best_auc_acc": best_auc_acc,
        "best_auc": best_auc,
        "best_val_loss": best_val_loss
    }
    torch.save(save_dict, os.path.join(fold_output_dir, args.name+ str("_final_checkpoint.pth.tar")))
    return
def main(args):
    # https: // stackoverflow.com / questions / 63221468 / runtimeerror - dataloader - worker - pid - 27351 - is -killed - by - signal - killed
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    # Training settings
    now = str(datetime.now())
    # init_distributed_mode()

    # From: https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.model == "3DCNN":
        torch.use_deterministic_algorithms(mode=False) #mode false needed for max_pool3d_with_indices_backward_cuda
        set_determinism(args.seed, use_deterministic_algorithms=False)  # MONAI
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        set_determinism(args.seed, use_deterministic_algorithms=True)  # MONAI

    # https://pytorch.org/docs/stable/notes/randomness.html
    g = torch.Generator()  # Needed for reproduciblility
    g.manual_seed(args.seed)
    name_org = args.name
    if args.cross_validate:
        for k in range(args.k_start, args.k_end):
            args.name = name_org + "_fold" + str(k)  # Add fold number to group
            print("Starting training for fold" + str(k))
            init_wandb(args, k, group=args.name, name=args.name)
            train_fold(k, args, now, g)
            wandb.finish()
    else:
        k = args.k
        args.name = name_org + "_fold" + str(k)
        init_wandb(args, args.k, group=args.name, name=args.name)
        train_fold(args.k, args, now, g)
        wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

# https://github.com/pytorch/pytorch/issues/1355#issuecomment-907986126 ??