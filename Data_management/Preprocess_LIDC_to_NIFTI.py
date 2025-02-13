
# dcm2niix -o /home/eloy/Desktop/LUNG_DATA/NIFTI/ -z y /home/eloy/Desktop/LUNG_DATA/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/


# Since there is multiple annotation classes per nodule (one per radiologist), you first want to loook for the scan, then the nodule, and then match the annotations.
import pylidc as pl
from pylidc.utils import consensus
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split
OUTPUT_DIR = "/home/eloy/Documents/Data/LUNG_DATA/"
NIFTI_OUTPUT_DIR = os.path.join(OUTPUT_DIR,"NIFTI/")


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_nifti(array, filename, folder, spacing):
    mkdir(folder)
    array = np.transpose(array, (2, 1, 0))
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image.SetSpacing(spacing)  # Replace with the appropriate voxel spacing (e.g., (0.5, 0.5, 1.0))
    sitk_image.SetDirection([0., 1., 0.,
                             1., 0., 0.,
                             0., 0., 1.])
    sitk.WriteImage(sitk_image, os.path.join(folder, filename))
def process_scan(scan_id):
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == scan_id).first()
    nods = scan.cluster_annotations()
    mypad = [(1000, 1000), (1000, 1000), (1000, 1000)]
    image_array = scan.to_volume(verbose=False)
    image_name = scan.patient_id
    pixel_spacing = scan.pixel_spacing
    spacing = (scan.pixel_spacing, scan.pixel_spacing, scan.slice_thickness)

    found_valid_nodule = False

    data = {
        'Tumor ID': [],
        'Image path': [],
        'Annotation path': [],
        'Label': []
    }
    df = pd.DataFrame(data)
    df_fast = pd.DataFrame(data)
    for i, nod in enumerate(nods):
        if len(nods[i]) > 3:  # 4 or more radiologists have annotated this data.
            malignancies = []
            for ann in nod:
                malignancies.append(ann.malignancy)
            malignancy = np.median(malignancies)

            if malignancy == 3:
                continue
            elif malignancy < 3:
                label = "B"
            elif malignancy > 3:
                label = "M"
            cmask, cbbox, masks = consensus(nod, clevel=0.5, pad=mypad)
            if np.sum(cmask) < 10:
                print("Too small tumor for " + str(image_name) + "... skipping")
                continue


            cropped_img = np.clip(image_array, -1000, 400) #https://www.researchgate.net/publication/369865428_CT_Lung_Nodule_Segmentation_A_Comparative_Study_of_Data_Preprocessing_and_Deep_Learning_Models#pf2
            # Normalize the array between 0 and 1
            min_val = np.min(cropped_img)
            max_val = np.max(cropped_img)
            cropped_img = (cropped_img - min_val) / (max_val - min_val)
            if np.sum(cropped_img*cmask) == 0:
                raise ValueError("Product of mask and image is zero for " + str(image_name))
            if (np.sum(cmask.astype(int)) == 0):
                print("SUM",np.sum(cmask))
                print("SUM_INT",np.sum(cmask.astype(int)))
                raise ValueError("mask is zero for " + str(image_name))

            if np.shape(cmask) != np.shape(image_array):
                print(np.shape(cmask))
                print(np.shape(image_array))
                raise ValueError("Not same shape for mask and image " + str(image_name))
            tumor_height = len(np.unique(np.nonzero(cmask)[-1])) * spacing[-1]
            if tumor_height < 3:
                print("Removing "+str(image_name)+"because the tumor is less than 3 mm in height")
                continue

            if True==True:
                target_folder = os.path.join(NIFTI_OUTPUT_DIR, image_name)
                mask_path_name = label + "_" + image_name + "_" + str(i) + ".nii.gz"
                image_path_name = image_name + ".nii.gz"
                df = df.append({
                    'Tumor ID': image_name + "_" + str(i),
                    'Image path': os.path.join(image_name, "IMAGES", image_path_name),
                    'Annotation path': os.path.join(image_name, "MASKS", mask_path_name),
                    'Label': label
                }, ignore_index=True)
                #If we hadn't previously saved the image, we will save the image here as we now have a valid annotation.
                if not found_valid_nodule:
                    found_valid_nodule = True
                    mkdir(target_folder)
                    save_nifti(image_array, image_path_name, os.path.join(target_folder, "IMAGES"), spacing)
                save_nifti(cmask.astype(int), mask_path_name,
                           os.path.join(target_folder, "MASKS"), spacing)
    if found_valid_nodule:
        return df, df_fast
    else:
        return None
if __name__ == "__main__":
    data = {
        'Tumor ID': [],
        'Image path': [],
        'Annotation path': [],
        'Label': []
    }
    df = pd.DataFrame(data)
    df_fast = pd.DataFrame(data)
    ha = 0;
    l=len(list(pl.query(pl.Scan)))

    scan_list = set([a.patient_id for a in list(pl.query(pl.Scan))])
    results=process_map(process_scan, scan_list, max_workers=20,chunksize=1)

    for result in tqdm(results, desc="Converting LIDC to NIFTI",unit="item"):
        if isinstance(result,type(None)):
            continue
        r, r_fast = result
        df=df.append(r,ignore_index=True)
        df_fast = df_fast.append(r_fast, ignore_index=True)
    print("Total usable lesions:" + str(len(df)))

    # For compatibility with our older methods, we make a data split here. Note that this is NOT used in the current code, as we use nested cross validation which creates its own test splits!
    print("Creating train/test")
    # Splitting the DataFrame into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'])
    # Adding a column to indicate whether a sample is in the test set or not
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['Internal testset'] = 0
    test_df['Internal testset'] = 1
    # Combining train and test sets back into one DataFrame
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.copy()
    df['Internal testset tiny'] = df['Internal testset']
    df['External testset'] = 0
    df['Inclusion'] = 1
    df['outlier'] = 0
    df['Center'] = "None"
    df['Patient ID'] = range(1, len(df) + 1)
    csv_file = os.path.join(NIFTI_OUTPUT_DIR,'dataset.csv')
    df.to_csv(csv_file, index=False)


    print("DataFrame successfully saved to CSV:", csv_file)