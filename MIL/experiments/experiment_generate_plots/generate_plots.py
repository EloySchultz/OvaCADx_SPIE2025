import wandb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import pickle
import matplotlib.lines as mlines

wandb.login()


# Example lists of run IDs (replace these with your actual run IDs)
run_ids_1 = ["rwde972b",  #Ovarian baseline
           "lbj5o59h",
           "dhcjhc14",
           "onj9cvh4",
           "74kx9qtn",
           "x6h6lrj7",
           "znfuwolb",
           "aqqadalm",
           "ki73iz9j",
           "h0c8ug2i",
           "0vd724eq",
           "ugqclnqz",
           "ui2bz9a0",
           "5eeada5p",
           "cyee0daq",
           "2go5iomy",
           "5068qrro",
           "psr8hxps",
           "en5psg3o",
           "0t8r6zad",
           "ksfv3l4t",
           "z8pmbxw2",
           "w939k54g",
           "j2vbmo1j",
           "siehxdaj"
           ]
run_ids_2 = ["xpafumo6", #3dcnn ovarian
           "1zmwijdf",
             "yb6kbuyk",
             "8r3ylfrn",
             "fmqo1o9e",
             "wt4xz68m",
             "3q3at3n6",
             "giew11d8",
             "yc7ptxkp",
             "63rxac7x",
             "hlpeutsu",
             "sfhw6q21",
             "3z27imnf",
             "17q93b1z",
             "4zzhzvtl",
             "f1uj8gxu",
             "3hs3nfsx",
             "fcuailpz",
             "a2j5kra5",
             "j2vcrqx7",
             "qv678tif",
             "s2lfh5qc",
             "7s3frsic",
             "mzfa2gfe",
             "m1c8u7f2"
           ]
run_ids_3 = ["x01exzwe",
             "ncwpvpit",
             "wbzl27xt",
             "7dr4xujl",
             "ywhcq3xw",
             "sxyf1tsr",
             "oq0titmp",
             "dnjnceze",
             "ghyqrfks",
             "c6rbxf9i",
             "r41ru9ta",
             "gi1no77h",
             "9q6us8sc",
             "splyxgt9",
             "7zfqdubs",
             "4xovxqya",
             #"i7e6m6gi", #This iteration of fold 16 crashed. Need to retrain
             "5cprni3z",
             "ts1vjpuh",
             "pl7n8l5h",
             "1lnh5mcm",
             "6gt9m1vf",
             "3cgtrm4j",
             "gzsxve56",
             "scks2nki"

           ]

run_ids_4 = ["xkm9gxuv", #3dcnn LIDC
           "l17xh4hd",
             "zk6fl7bl",
             "xzukf0kc",
             "yg1hnj3t",
             "ckydqngi",
             "drufguxg",
             "79kxz58w",
             "k0mln0so",
             "2inhysp8",
             "2t1mlwiu",
             "abc58jlt",
             "4yem58rq",
             "jyhiyww0",
             "njupuisv",
             "v5lvlbvn",
             "ujg33daz",
             "emfbqvhk",
             "lievmv96",
             "1ou33o61",
             "c11jy849",
             "94m5mffo",
             "iod70btu",
             "syvlrq1t",
             "hd0p17wb"

           ]
names = ["MILCNN","3DCNN"]


# Function to retrieve data and calculate the mean, min, and max for a list of run IDs
def get_mean_metrics(run_ids, data_dir="metrics_data", force_download=False):
    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Initialize lists for storing metrics
    train_loss_list = []
    val_loss_list = []
    train_auc_list = []
    val_auc_list = []
    steps_list = []

    # Retrieve data for each run
    for run_id in tqdm(run_ids, desc="Processing runs"):
        file_path = os.path.join(data_dir, f"{run_id}.pkl")

        # Check if the file exists and if force_download is False
        if os.path.exists(file_path) and not force_download:
            with open(file_path, "rb") as file:
                run_data = pickle.load(file)
        else:
            # If not on disk or force_download is True, download the data
            run = wandb.Api().run(f"eloyschultz1999/MIL CADX ELOY/{run_id}")
            history = run.history(keys=["Train loss", "Val loss", "Train auc", "Val auc", "_step"])

            run_data = {
                'train_loss': history["Train loss"].values,
                'val_loss': history["Val loss"].values,
                'train_auc': history["Train auc"].values,
                'val_auc': history["Val auc"].values,
                'steps': history["_step"].values
            }

            # Save the data to disk
            with open(file_path, "wb") as file:
                pickle.dump(run_data, file)

        # Append the run data to the corresponding lists
        train_loss_list.append(run_data['train_loss'])
        val_loss_list.append(run_data['val_loss'])
        train_auc_list.append(run_data['train_auc'])
        val_auc_list.append(run_data['val_auc'])
        steps_list.append(run_data['steps'])

    # Convert lists to numpy arrays
    train_loss_array = np.array(train_loss_list)
    val_loss_array = np.array(val_loss_list)
    train_auc_array = np.array(train_auc_list)
    val_auc_array = np.array(val_auc_list)
    steps_array = np.array(steps_list)

    # Calculate means, mins, and maxs
    mean_train_loss = np.mean(train_loss_array, axis=0)
    mean_val_loss = np.mean(val_loss_array, axis=0)
    mean_train_auc = np.mean(train_auc_array, axis=0)
    mean_val_auc = np.mean(val_auc_array, axis=0)
    min_train_loss = np.percentile(train_loss_array,25, axis=0)
    max_train_loss = np.percentile(train_loss_array,75, axis=0)
    min_val_loss = np.percentile(val_loss_array,25, axis=0)
    max_val_loss = np.percentile(val_loss_array,75, axis=0)
    min_train_auc = np.percentile(train_auc_array,25, axis=0)
    max_train_auc = np.percentile(train_auc_array,75, axis=0)
    min_val_auc = np.percentile(val_auc_array,25, axis=0)
    max_val_auc = np.percentile(val_auc_array,75, axis=0)
    steps = steps_array[0]

    return (steps, mean_train_loss, mean_val_loss, mean_train_auc, mean_val_auc,
            min_train_loss, max_train_loss, min_val_loss, max_val_loss,
            min_train_auc, max_train_auc, min_val_auc, max_val_auc)

# Create subplots: 4 columns, 2 rows
fig, axes = plt.subplots(2, 4, figsize=(14, 6))

column_titles = ['Ovarian', 'LIDC']

g=14
# Iterate over each set of run IDs and plot the results
for i, run_ids in enumerate([run_ids_1, run_ids_2, run_ids_3, run_ids_4]):
    (steps, mean_train_loss, mean_val_loss, mean_train_auc, mean_val_auc,
     min_train_loss, max_train_loss, min_val_loss, max_val_loss,
     min_train_auc, max_train_auc, min_val_auc, max_val_auc) = get_mean_metrics(run_ids)

    # Top plot: mean Train loss and Val loss with shaded area
    ax_loss = axes[0, i]  # Upper row for loss
    ax_loss.plot(steps, mean_train_loss, label='T Loss', color='blue')
    ax_loss.plot(steps, mean_val_loss, label='V Loss', color='orange')
    ax_loss.fill_between(steps, min_train_loss, max_train_loss, color='blue', alpha=0.2)
    ax_loss.fill_between(steps, min_val_loss, max_val_loss, color='orange', alpha=0.2)
    ax_loss.set_xlabel('Epoch', fontsize=g)
    ax_loss.set_ylabel('Loss',fontsize=g)
    ax_loss.set_title(f'Loss for {names[i % 2]}',fontsize=g)
    ax_loss.legend(fontsize=13)
    ax_loss.grid(True)

    # Bottom plot: mean Train auc and Val auc with shaded area
    ax_auc = axes[1, i]  # Lower row for AUC
    ax_auc.plot(steps, mean_train_auc, label='T AUC', color='green')
    ax_auc.plot(steps, mean_val_auc, label='V AUC', color='red')
    ax_auc.fill_between(steps, min_train_auc, max_train_auc, color='green', alpha=0.2)
    ax_auc.fill_between(steps, min_val_auc, max_val_auc, color='red', alpha=0.2)
    ax_auc.set_xlabel('Epoch',fontsize=g)
    ax_auc.set_ylabel('AUC', fontsize=g)
    ax_auc.set_title(f'AUC for {names[i % 2]}', fontsize=g)
    ax_auc.legend(fontsize=13)
    ax_auc.grid(True)

    # Set sharey only for the bottom row
    if i > 0:
        ax_auc.sharey(axes[1, 0])

    # Make sure all axes have ticks
    ax_loss.tick_params(axis='both', which='both', direction='in', length=5)
    ax_auc.tick_params(axis='both', which='both', direction='in', length=5)

    # Add subplot identification letters
    #label = chr(65 + 2*i)  # Alphabet label for top plot (A, C, E, G)
    N = 65 + i
    label = chr(N)
    ax_loss.text(-0.1, 1.1, label, transform=ax_loss.transAxes,
                 fontsize=18, fontweight='bold', va='top', ha='right',fontfamily="Times New Roman")
    label = chr(N+4)
    ax_auc.text(-0.1, 1.1, label, transform=ax_auc.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right',fontfamily="Times New Roman")

# Add extra titles above each pair of columns
for i, title in enumerate(column_titles):
    fig.text(0.27 + i*0.49, 0.95, title, ha='center', va='center', fontsize=18, font="DejaVu Sans")

# Add a vertical divider line between the 4 left and 4 right plots
divider_x = 0.502  # Position for the vertical divider (in figure coordinates)
divider = mlines.Line2D([divider_x, divider_x], [0, 1], transform=fig.transFigure, color="black", linestyle="--", linewidth=1)
fig.add_artist(divider)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.93])

plt.savefig("Training curves.pdf")
# Show the plot
plt.show()
