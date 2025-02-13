import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def plot_volume_distributions_with_fit(paths, names,output_path):
    """
    Plot frequency bar charts with Gaussian fits for tumor volumes from multiple CSV files, using a logarithmic x-axis.
    The plots are arranged in a 2x2 grid.
    A green line indicates the volume threshold that divides the distribution into two equal parts.

    Parameters:
    - paths (list): List of directory paths containing `preprocessed_data.csv`.
    - output_path (str): Path to save the generated subplot image.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex='col')  # 2x2 subplot layout, sharing x-axis by column
    axes = axes.ravel()  # Flatten the 2D array of axes into a 1D array

    for i, path in enumerate(paths):
        name=names[i]
        csv_file = os.path.join(path, 'preprocessed_data.csv')

        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            continue

        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Filter data based on conditions: 'inclusion' == 1 and 'label' != 'BL'
        df_filtered = df[(df['label'] != 2) & (df['center'] != "AVL")]

        # Ensure the 'volume_cm3' column exists
        if 'volume_cm3' not in df_filtered.columns:
            print(f"'volume_cm3' column missing in {csv_file}")
            continue

        # Extract the volumes
        volumes = df_filtered['volume_cm3']

        # Filter out zero or missing values to focus on actual tumor volumes
        volumes = volumes[volumes > 0]

        if volumes.empty:
            print(f"No valid tumor volumes in {csv_file}")
            continue

        # Convert volumes to log scale
        log_volumes = np.log10(volumes)

        # Define the bins for the histogram
        bins = np.linspace(min(log_volumes), max(log_volumes), 30)

        # Determine the label for filtering the subplots
        for label in [0, 1]:
            # Filter the data based on label
            df_label = df_filtered[df_filtered['label'] == label]
            volumes_label = df_label['volume_cm3']
            volumes_label = volumes_label[volumes_label > 0]

            if volumes_label.empty:
                continue

            log_volumes_label = np.log10(volumes_label)

            # Select subplot axes based on the label and path index
            if i == 0:  # path_1
                if label == 0:
                    ax = axes[0]  # Top-left
                else:
                    ax = axes[2]  # Bottom-left
            elif i == 1:  # path_2
                if label == 0:
                    ax = axes[1]  # Top-right
                else:
                    ax = axes[3]  # Bottom-right

            # Plot histogram for the current label
            ax.hist(log_volumes_label, bins=bins, density=True, color="skyblue", alpha=0.6, edgecolor="black", label="Histogram")

            # Fit a Gaussian distribution to log-transformed volumes
            mean, std = norm.fit(log_volumes_label)

            # Generate x values for the fitted curve (log scale)
            x = np.linspace(min(log_volumes_label), max(log_volumes_label), 1000)
            y = norm.pdf(x, mean, std)

            # Plot the Gaussian fit
            ax.plot(x, y, color="red", linestyle="--", label=f"$\\mu={mean:.2f}, \\sigma={std:.2f}$")

            # Calculate the median (threshold) on the log scale
            median = np.median(log_volumes_label)

            # Add a green vertical line at the threshold
            ax.axvline(x=median, color='green', linestyle='-', label=f'Median = {10 ** median:.2e} cm³')

            # Set title, labels, and legend
            ax.set_xlabel(r'$\log_{10}(\mathrm{Volume}) \, [\log(\mathrm{cm}^3)]$', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend()
            if label==0:
                lbl="Benign"
            else:
                lbl="Malignant"
            if i == 0:
                ax.set_title(f"Volume distribution for {name}, {lbl}", fontsize=10)
            else:
                ax.set_title(f"{name}, {lbl}", fontsize=10)

    # Adjust layout
    plt.tight_layout()
    fig.suptitle("Tumor Volume Distributions with Gaussian Fits (Log Scale)", fontsize=16, y=1.02)

    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Frequency bar charts with Gaussian fits saved to {output_path}")


def main():
    names = ["Ovarian","LIDC"]
    paths = [
        "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4-Processed_new_FAST",
        "/home/eloy/Documents/Data/LUNG_DATA/NIFTI-Processed_new_FAST"
    ]

    output_path = "./volume_distributions_2x2_grid.png"
    plot_volume_distributions_with_fit(paths,names, output_path)


if __name__ == "__main__":
    main()
