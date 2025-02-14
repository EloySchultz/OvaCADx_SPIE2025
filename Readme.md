# Identifying Key Challenges in Ovarian Tumor Classification: A Comparative Study Using Deep Learning and Radiomics

This repository contains the code for the SPIE publication titled "Identifying Key Challenges in Ovarian Tumor Classification: A Comparative Study Using Deep Learning and Radiomics".

## Index

1. [Interpreters](#1-interpreters)
2. [Dataset Preprocessing](#2-dataset-preprocessing)
    - [Ovarian Dataset](#21-ovarian-dataset)
    - [LIDC Dataset](#22-lidc-dataset)
3. [Extracting Radiomic Features](#3-extracting-radiomic-features)
4. [Deep Learning Instructions](#4-deep-learning-instructions)
5. [Reproducing Results from the SPIE Publication](#5-reproducing-results-from-the-spie-publication)
6. [Model Training Instructions](#6-model-training-instructions)
7. [Footnote on AUC_median](#7-footnote-on-auc_median)

## 1. Interpreters

To run our code, two interpreters are required: "GP_1" and "GP_3". The corresponding requirements for each interpreter, including packages and versions, are found in `requirements_GP1.txt` and `requirements_GP3.txt`. By default, use the GP_1 interpreter for all training and inference tasks. The GP_3 interpreter is only needed for preprocessing the LIDC dataset, as PyLIDC relies on an outdated version of numpy. We recommend using the versions specified in the `requirements.txt` files.

## 2. Dataset Preprocessing

It is recommended to open the project in the `data_management` folder (i.e., set `/Data_management/` as the root directory).

### 2.1 Ovarian Dataset

Collect your ovarian data and create a `datasets.csv` file. The root directory must contain the `datasets.csv` with the following columns:

1. **Patient ID**: (Int) Unique identifier for each patient.
2. **Tumor ID**: (String) Unique identifier for each tumor (starts with center abbreviation, in our CZE, BRE, or AVL in our case.).
3. **Center**: (String) The center where the scan was done (CZE, BRE, or AVL).
4. **Image path**: (String) Path to image nifti file, ending in `.nii.gz`.
5. **Annotation path**: (String) Path to annotation nifti file, ending in `.nii.gz`.
6. **Label**: (String) Label (B: benign, M: malignant, BL: borderline).
7. **Inclusion**: (Int) Whether the sample is included in the research (1 or 0).
8. **Outlier**: (Int) Whether the sample is an easy outlier, such as dermoids (1 for outlier, 0 for non-outlier). This is not used in the SPIE publication and can be skipped.

For my colleagues at the TU/e, this is dataset 3.4 from the Flux-server at SPS-VCA.
#### Required Steps:
- Run the preprocessing script:   
  `/ovacadx/scripts/preprocess_data.py --data_path "Path to original dataset with datasets.csv" --output_path "Path to store processed files"`.  
  This generates a `preprocessed_data.csv` in the output path, resampling images to 3mm when needed.

- Optional error checking:  
  Run `/Data_management/Preprocess_ovarian_boekhouding.py` for more sophisticated error checking.
  
- Optional manual review:  
  Use `/Data_management/Manual_data_check.py` for manually reviewing annotation masks (interactive video controls for quality assurance). 

- Required:  
  Crop the ovarian dataset around tumors for faster disk reads during training (from 30.5GB to approx 1.3GB):  
  Run `Data_management/Create_faster_ovarian.py` after setting `DATA_DIR` to the preprocessed folder.

### 2.2 LIDC Dataset

For comparison, we also use the public LIDC IDRI lung nodule dataset. Preprocess it to match the format of the ovarian dataset (use the GP_3 interpreter for these steps).

#### Required Steps:
- Download the LIDC dataset:  
  Use the *NBIA Data Retriever* tool from [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254). Install it using `sudo dpkg -i nbia-data-retriever-4.4.2.deb`. Make sure to enable **Classic Directory Name** when downloading the dataset. 

- Install PyLIDC:  
  Follow the installation guide for PyLIDC at [this link](https://pylidc.github.io/install.html).

- Convert LIDC dataset to NIFTI format:  
  Use the script `/Data_management/Preprocess_LIDC_to_NIFTI.py`.

- Standardize and crop LIDC NIFTI dataset:  
  Run `/ovacadx/scripts/preprocess_data.py` on the NIFTI files, then use `Data_management/Create_faster_ovarian.py` to create the cropped dataset.

After processing both datasets, you will have the following dataset folders:

1. `Ovarian/NIFTI`
2. `Ovarian/NIFTI_FAST`
3. `LIDC/NIFTI`
4. `LIDC/NIFTI_FAST`

Where `_FAST` datasets contain only cropped tumor regions for faster loading.

## 3. Extracting Radiomic Features

To work with radiomics, open the project in the radiomics folder (`/ovacadx/`) as the root directory.

#### Required Steps:
- Extract radiomic features:  
  Run `/Code/ovacadx/scripts/Radiomics/extract_features.py`, which creates a `3DRadiomicfeatures.json`.

- Radiomics workflow:
  1. Train models:  
     Run `/Code/ovacadx/scripts/Radiomics/train.py` (use 25-fold nested cross-validation).
  2. Test models:  
     Run `/Code/ovacadx/scripts/Radiomics/test.py` on the training results.
  3. Ensemble results:  
     Run `/Code/ovacadx/scripts/Radiomics/Analysis/ens_no_logits2.py` to apply bootstrapping and generate compiled results.

For reproducibility, combine these steps in one script. An example is `/Code/ovacadx/scripts/Radiomics/Experiment_radiomics_SPIE/Experiment_SPIE.py`.

### 3.1 Feature Selection Pipeline in Radiomics

The feature selection pipeline is controlled by the `--pipeline` and `--features` arguments in the `train.py` and `test.py` scripts.

#### Example:
- Run:  
  `--pipeline SELECT_PERCENTILE,PS,BLACK_BOX --features 10,40,0`
  This will:
  - Select 10% of features using percentile ranking.
  - Retain 40 features based on Pearson-Spearman correlation.
  - Apply Random Forest-based feature elimination.

## 4. Deep Learning Instructions

For deep learning, the order of operations is the same as for radiomics:  
`train.py` -> `test.py` -> `ens_no_logits2.py`. 
See `/MIL/experiments/experiment_ovarian_baseline/Ovarian_baseline.py` for an example.
Both radiomics and deep learning analyses use the same bootstrapping approach by using the same `ens_no_logits2.py`. 

## 5. Reproducing Results from the SPIE Publication

Make sure all previous sections are completed. Follow these steps to reproduce figures and tables from the publication:

- **Figure 1**: Run `/MIL/experiments/experiment_show_tumors/show_tumors.py`.
- **Table 1**: Run `/ovacadx/scripts/Radiomics/Experiment_statistics/statistics.py`.
- **Figure 2 & 3**: Created in Adobe Illustrator.
- **Figure 4**: Run `/ovacadx/scripts/Radiomics/Experiment_plot_ks_SPIE/Experiment_plot_ks.py`.
- **Table 2**: After training, testing, and ensembling, run `/ovacadx/scripts/Radiomics/Experiment_score_table_2_SPIE/score_table.py`.
- **Figure 5**: After training, testing, and ensembling, run `/ovacadx/scripts/Radiomics/Analysis/Half_subset_auc.py`.

## 6. Model Training Instructions

Follow these steps to train the models presented in the publication (5 models on 2 datasets):

1. Run `/ovacadx/scripts/Radiomics/Experiment_radiomics_SPIE/Experiment_SPIE.py`. (This generates all radiomics-based results for both datasets, and for multiple classifiers. Only the NN classifier result is used in the SPIE publication). 
2. Run `/MIL/experiments/experiment_ovarian_baseline/Ovarian_baseline.py`.
3. Run `/MIL/experiments/experiment_lung_baseline/exp_lung_baseline.py`.
4. Run `/MIL/experiments/experiment_3DCNN_ovarian/exp_3DCNN_ovarian_baseline.py`.
5. Run `/MIL/experiments/experiment_3DCNN_LUNG/experiment_3DCNN_LUNG.py`.

Enable rotation augmentation:

6. Run `/MIL/experiments/experiment_ov_rotate/Ov_rotate.py`.
7. Run `/MIL/experiments/experiment_lung_rotate/lung_rotate.py`.
8. Run `/MIL/experiments/experiment_3DCNN_OvarianRotate/3dcnn_ovarian_rotate.py`.
9. Run `/MIL/experiments/experiment_3DCNN_LIDCRotate/3dcnn_lung_rotate.py`.

Ensure to uncomment `#os.system(command)` lines for retraining (We have uncommented the training here and there for convenience during evaluation).

## 7. Footnote on AUC_median

The AUC_median for all radiomic features is available at `/ovacadx/scripts/Radiomics/Analysis/Optimal_half_AUCs_per_feature.csv`.
These values are calculated using `/ovacadx/scripts/Radiomics/Analysis/Half_subset_auc.py`.

## Disclaimer

This code was copied from a larger internal repository, which also includes development files. If any files are missing, please report the issue.
