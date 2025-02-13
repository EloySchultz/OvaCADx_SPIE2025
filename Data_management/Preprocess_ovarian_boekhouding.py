import SimpleITK as sitk
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
import os
import nibabel as nib
import shutil

#Checks for all kinds of errors and corrects them.
# Multicomponents in single nifti file
# Cropped masks/images
# Nibabel incompatibility
# Inconsistent z-spacing (resamples to 3mm if needed)
# Reports when mask in place of image or vice versa


#Specify root directory of dataset
dataset_directory = "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4/"
#Specify path to boekhouding excelsheet
file_path = "/home/eloy/Documents/Data/OVARY_DATA/Data_gyn_3.4/Excel files/Combi_boekhouding_vs270224.xlsx"




def check_data(dataset_directory,inclusion_df):
    save_df = pd.DataFrame(columns=["ID","Path", "Remark"])

    if not os.path.exists(dataset_directory) or not os.path.isdir(dataset_directory):
        raise ValueError(f"The directory '{dataset_directory}' does not exist.")

    h = 0;
    mask_paths = []
    # First, we replace all .nii files with .nii.gz
    nii_files=[]
    for filename in glob.glob(dataset_directory + '/**/*.nii', recursive=True):
        niipath = os.path.join(dataset_directory,filename)
        nii_files.append(niipath)
    for file in nii_files:

        # Replace 'input.nii' with the path to your original .nii file
        input_nii_file = file
        output_nii_gz_file = input_nii_file.rsplit('.', 1)[0] + ".nii.gz"
        img = sitk.ReadImage(input_nii_file)
        sitk.WriteImage(img, output_nii_gz_file)
        os.remove(input_nii_file)

        print(f"{input_nii_file} has been compressed to {output_nii_gz_file}, and the original file has been deleted.")

    image_paths = inclusion_df['Image_Path']
    mask_paths = inclusion_df['Annotation_Path']
    labels = inclusion_df['Label']
    IDs = inclusion_df['Tumor ID']

    #Please note that we do some checkin on the test set. This is not used in the current version of our code, as we switched to using nested crossvalidation which creates its own test sets.
    inclusion_df['Nieuwe test set'] = inclusion_df['Nieuwe test set'].fillna(0.0)
    # Change "Nieuwe test set" to 1 for rows where "Centre" is "AVL"
    inclusion_df.loc[inclusion_df['Centre'] == 'AVL', 'Nieuwe test set'] = 1
    test_set_yn = inclusion_df['Nieuwe test set']
    external_test_set_yn = (inclusion_df['Centre'] == 'AVL').tolist()
    non_existent_files = []
    # Check if each file exists
    for file_path in image_paths:
        file_path = os.path.join(dataset_directory,file_path)
        if not os.path.exists(file_path):
            non_existent_files.append(file_path)
    for file_path in mask_paths:
        file_path = os.path.join(dataset_directory, file_path)
        if not os.path.exists(file_path):
            non_existent_files.append(file_path)
    if len(non_existent_files)>0:
        print("The following file paths are incorrect:")
        print(len(non_existent_files))
        print(non_existent_files)

    print("We will check: "+ str(len(image_paths)) + " files...")
    data = {
        'ID': IDs,
        'Image_Path': image_paths,
        'Annotation_Path': mask_paths,
        'Label': labels,
        'Test_set': test_set_yn,
        'External_test_set': external_test_set_yn
    }
    df = pd.DataFrame(data)

    l = len(df)
    with tqdm(total=l, desc="Checking data") as pbar:
        for index, row in df.iterrows():
            pbar.update(1)
            img = os.path.join(dataset_directory, row['Image_Path'])
            msk = os.path.join(dataset_directory, row['Annotation_Path'])
            ## Fix for https://stackoverflow.com/questions/74039929/nibabel-cannot-read-gz-file
            try:
                nib_t1=nib.load(img)
            except:

                sitk_t1 = sitk.ReadImage(img)
                sitk.WriteImage(sitk_t1, img)
                try:
                    nib_t1 = nib.load(img)
                except:
                    raise ValueError("ERR: NiBabel error keeps occuring on "+ str(img))
                print("Fixed nibabel error on " + str(img))
            try:
                mask_nib_t1 = nib.load(msk)
            except:

                sitk_t1 = sitk.ReadImage(msk)
                sitk.WriteImage(sitk_t1, msk)
                try:
                    mask_nib_t1 = nib.load(msk)
                except:
                    raise ValueError("ERR: NiBabel error keeps occuring on " + str(msk))
                print("Fixed nibabel error on " + str(msk))

            sitk_t1 = sitk.ReadImage(img)
            mask_sitk_t1 = sitk.ReadImage(msk)

            if sitk_t1.GetSpacing() != mask_sitk_t1.GetSpacing():
                #print("WARN: Image and mask have different spacing: "+img)
                #This fix is needed for CZE_B01 specifically, as its metadata is incoherent with the actual spacing of the data.
                if sitk_t1.GetSize() == mask_sitk_t1.GetSize(): #Size is same but spacing is different, hence spacing of mask should be adjusted to be correct again (no resampling needed).
                    sform = nib_t1.get_qform() #Copy sform from image.
                    mask_nib_t1.set_sform(sform)
                    mask_nib_t1.set_qform(sform)
                    mask_nib_t1.header['sform_code']=0
                    directory_path, filename = os.path.split(msk)
                    name, extension = os.path.splitext(os.path.splitext(filename)[0])
                    path_without_extension = os.path.join(directory_path, name)
                    # Save the resampled image and keep backup of original
                    shutil.copyfile(msk, path_without_extension + "_OLD_WRONG_METADATA.nii.gz") #Save old one for backup purposes.
                    nib.save(mask_nib_t1,msk)
                    print("Fixed: Non-matching MASK IMAGE PAIR: " + str(img))
                else:
                    print("ERR: Non-matching MASK IMAGE PAIR: " + str(img))
                    df.drop(index=index, inplace=True)
                    continue

            t1 = sitk.GetArrayFromImage(sitk_t1)
            if len(t1.shape) > 3:
                # Check if multiple components contain same data. If so, we should just remove one of the components.
                if (np.array_equal(t1[:, :, :, 0], t1[:, :, :, 1])):
                    print("Fixed: removed extra component from:", img)
                    t2 = t1[:, :, :, 0]
                    imz = sitk.GetImageFromArray(t2)
                    imz.CopyInformation(sitk_t1)
                    # write the image
                    sitk.WriteImage(imz, img)
                    sitk_t1 = sitk.ReadImage(img)
                    t1 = sitk.GetArrayFromImage(sitk_t1)
                else:
                    print("ERR: Ambiguous multicomponent for " + img)
                    df.drop(index=index, inplace=True)
                    continue

            # Check for slice_thickness and resample if thickness is incorrect.
            if not (sitk_t1.GetSpacing()[-1]<3.1 and sitk_t1.GetSpacing()[-1]>2.9):
                original_z = sitk_t1.GetSpacing()[-1]
                new_spacing = [sitk_t1.GetSpacing()[0], sitk_t1.GetSpacing()[1],
                               3.0]  # Keep x and y spacing unchanged, set z spacing to 3.0
                new_size = [int(round(s * os / ds)) for s, os, ds in
                            zip(sitk_t1.GetSpacing(), sitk_t1.GetSize(), new_spacing)]


                resample = sitk.ResampleImageFilter()
                resample.SetInterpolator = sitk.sitkLinear
                resample.SetOutputDirection(sitk_t1.GetDirection())
                resample.SetOutputOrigin(sitk_t1.GetOrigin())
                resample.SetOutputSpacing(new_spacing)


                orig_size = np.array(sitk_t1.GetSize(), dtype=int)
                orig_spacing = sitk_t1.GetSpacing()
                resample.SetSize(new_size)
                resampled_image = resample.Execute(sitk_t1)
                directory_path, filename = os.path.split(img)

                # Split the filename into name and extension
                name, extension = os.path.splitext(os.path.splitext(filename)[0])

                # Concatenate the directory path with the filename without the extension
                path_without_extension = os.path.join(directory_path, name)
                # Save the resampled image
                shutil.copyfile(img, path_without_extension + "_original.nii.gz")
                sitk.WriteImage(resampled_image, img)
                print("Fixed: OLD_Z = "+str(original_z)+" | Resampled in Z-direction for  " + str(img))
                sitk_t1 = sitk.ReadImage(img)
                t1 = sitk.GetArrayFromImage(sitk_t1)

            if np.min(t1) > -10 or np.max(t1) < 100:
                # mask_inplace_of_image.append(i)
                print("ERR: mask in place of image for " + img)
                df.drop(index=index, inplace=True)
                continue
            # Check if image has 512x512 resolution (DEBATABLE). If not, then the image is likely cropped
            if (t1.shape[1] != 512) or (t1.shape[2] != 512):
                print("ERR: image cropped for " + img)
                df.drop(index=index, inplace=True)
                continue


            # Masks
            sitk_t1 = sitk.ReadImage(msk)
            t1 = sitk.GetArrayFromImage(sitk_t1)
            if len(t1.shape) > 3:
                # Check if multiple components contain same data. If so, we should just remove one of the components.
                if (np.array_equal(t1[:, :, :, 0], t1[:, :, :, 1])):
                    print("Fixed: removin extra component from:", msk)
                    t2 = t1[:, :, :, 0]
                    imz = sitk.GetImageFromArray(t2)
                    imz.CopyInformation(sitk_t1)
                    # write the image
                    sitk.WriteImage(imz, msk)
                    sitk_t1 = sitk.ReadImage(msk)
                    t1 = sitk.GetArrayFromImage(sitk_t1)
                else:
                    print("ERR: Ambiguous multicomponent for " + msk)
                    df.drop(index=index, inplace=True)
                    continue

            #reslice if Z-spacing is incorrect
            if not (sitk_t1.GetSpacing()[-1]<3.1 and sitk_t1.GetSpacing()[-1]>2.9):
                original_z = sitk_t1.GetSpacing()[-1]
                new_spacing = [sitk_t1.GetSpacing()[0], sitk_t1.GetSpacing()[1],
                               3.0]  # Keep x and y spacing unchanged, set z spacing to 3.0
                new_size = [int(round(s * os / ds)) for s, os, ds in
                            zip(sitk_t1.GetSpacing(), sitk_t1.GetSize(), new_spacing)]


                resample = sitk.ResampleImageFilter()
                resample.SetInterpolator = sitk.sitkLinear
                resample.SetOutputDirection(sitk_t1.GetDirection())
                resample.SetOutputOrigin(sitk_t1.GetOrigin())
                resample.SetOutputSpacing(new_spacing)


                orig_size = np.array(sitk_t1.GetSize(), dtype=int)
                orig_spacing = sitk_t1.GetSpacing()
                resample.SetSize(new_size)
                resampled_image = resample.Execute(sitk_t1)
                directory_path, filename = os.path.split(msk)

                # Split the filename into name and extension
                name, extension = os.path.splitext(os.path.splitext(filename)[0])

                # Concatenate the directory path with the filename without the extension
                path_without_extension = os.path.join(directory_path, name)
                # Save the resampled image
                shutil.copyfile(msk, path_without_extension + "_original.nii.gz")
                sitk.WriteImage(resampled_image, msk)
                print("Fixed: OLD_Z = "+str(original_z)+" | Resampled in Z-direction for  " + str(msk))
                sitk_t1 = sitk.ReadImage(msk)
                t1 = sitk.GetArrayFromImage(sitk_t1)
            if np.min(t1) < -10 or np.max(t1) > 100:
                # mask_inplace_of_image.append(i)
                print("ERR: image in place of mask for " + msk)
                df.drop(index=index, inplace=True)
                continue
            # Check if image has 512x512 resolution (DEBATABLE). If not, then the image is likely cropped
            if (t1.shape[1] != 512) or (t1.shape[2] != 512):
                print("ERR: mask cropped for " + msk)
                df.drop(index=index, inplace=True)
                continue


    df = pd.concat([df[df['Label'] != 'BL'], df[df['Label'] == 'BL']]) #Move BL to bottom of dataframe for maximum compatibility with previous artifact scans.
    duplicates = df[df.duplicated(subset='ID')]
    if not duplicates.empty:
        print("Duplicate files found:")
        print(duplicates)
    else:
        print("No duplicate files found.")

    print("Data check complete")
    print("Number of usable lesions: " + str(len(df)))
    print("Number of MALIGNANT lesions: " + str(len(df[df['Label'] == "M"])))
    print("Number of BENIGN lesions: " + str(len(df[df['Label'] == "B"])))
    print("Number of BORDERLINE lesions: " + str(len(df[df['Label'] == "BL"])))
    print("You can now create splits and preprocess nifti to H5 if you please. For creating splits, use Create_split.py")

    csv_file = os.path.join(dataset_directory, 'Samples.csv')
    df.to_csv(csv_file, index=False)
if __name__ == '__main__':
    df = pd.read_excel(file_path)
    inclusion_df = df[df['Inclusie '] == 1]  # Note de spatie
    #Create everything for ovarian data
    check_data(dataset_directory, inclusion_df)
