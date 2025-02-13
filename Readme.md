# Identifying Key Challenges in Ovarian Tumor Classification: A Comparative Study Using Deep Learning and Radiomics 

This repository details the code used for the SPIE-publication titled "Identifying Key Challenges in Ovarian Tumor Classification: A Comparative Study Using Deep Learning and Radiomics". 
# Index

1.  [Interpreters](#1-interpreters)
2.  [Dataset Preprocessing](#2-dataset-preprocessing)
    -   [Ovarian Dataset](#21-ovarian-dataset)
    -   [LIDC Dataset](#22-lidc-dataset)
3.  [Extracting Radiomic Features](#3-extracting-radiomic-features)
4.  [Deep Learning Instructions](#4-deep-learning-instructions)
5.  [Reproducing Results from the SPIE-publication](#5-How-to-reproduce-the-results-from-the-SPIE-publication?)
6.  [Model Training Instructions](#6-reproduction-of-our-model-trainings-for-tables)
7.  [Footnote on AUC_median](#7-Footnote-on-AUC_median)

# 1. Interpreters 
In order to run our code, there are two interpreters required. These are called "GP_1" and "GP_3".  The requirements for these environments, including packages and versions, can be found in requirements_GP1.txt and requirements_GP3.txt. The GP_1 interpreter should be used by default (for all training, inference, etc).  The GP_3 interpreter is used only for preprocessing the LIDC dataset. As of writing this, the PyLIDC library is outdated and uses numpy.int which is deprecated in newer versions of numpy. Hence the GP_3 interpreter uses an older version of Numpy. We recommend that you use the versions supplied in the requirements.txt documents for both interpreters.  


# 2. Dataset preprocessing
For the following steps, it is a good idea to open the project in the data_management folder. I.e., open the project with '/Data_management/' as the root directory. 
## 2.1 Ovarian dataset
Collect your data and create a datasets.csv file. (For us at TU/e, this is dataset 3.4 from the Flux-server) The root directory of the dataset must contain a file called datasets.csv, which contains at least the following columns for each sample:

1. Patient ID -> (Int) Unique identifier for each patient
2. Tumor ID --> (String) unique identifier for each tumor. Should start with CZE, BRE, or AVL based on  which center the scan came from
3. Center --> (String) which center the scan came from. Either CZE, BRE or AVL
4. Image path --> (String) path to image nifti file. Should end in .nii.gz. 
5. Annotation path --> (String)path to annotation nifti file. Shoud end in .nii.gz. 
6. Label --> (String) Label, either B, (benign) M (malignant) or BL (borderline)
7. Inclusion --> (Int) Whether or not sample should be included in this research. 1 or 0. 
8. outlier -->(Int) Whether the sample is an easy outlier, such as dermoids. 1 for outlier, 0 for no-outlier. 

- [**REQUIRED**] Run the preprocessing (`/ovacadx/scripts/preprocess_data.py --data_path
"Path to original dataset with datasets.csv"
--output_path
"Path where to store the processsed files"`)
In the output path, a file will be created called "preprocessed_data.csv". This script will resample the images to 3mm when needed and performs basic data-consistency checks. 

- [OPTIONAL] If your datasets contains inconsistencies, consider running: 
"/Data_management/Preprocess_ovarian_boekhouding.py" as it does more sophisticated error checking. 

- [OPTIONAL] If you want to manually check the quality of the annotation masks, run `/Data_management/Manual_data_check.py`. This file will play the CT-scans as videos for easy review. The controls are as follows: 
Space ( ): Toggles pause. Pressing it pauses and resumes the video.
S: Toggles slow motion mode, slowing down the playback speed.
B: Moves back to the previous sample without saving a label.
R: Reloads the current sample to watch again without changing labels.
G: Marks the current sample as "Good/OK, no artifacts" and assigns a usability grade of 5.
A: Marks the current sample as having artifacts. Prompts for a description and a usability grade (1–5).
I: Opens the current sample in ITKSNAP without modifying the label.
N: Moves to the next sample without altering the label.
ESC: Exits the program, saving progress up to that point.
H: Marks the sample as "Quality impeded by implant" with a usability grade of 4.


- [**REQUIRED**] We now will create yet another copy of the ovarian dataset, but now we will crop around the tumors and then save to the disk. This reduces the size of each NIFTI file such that reading from disk will be much faster during training. Doing this step reduces the size of the ovarian dataset from 30.5GB to approx 1.3 GB. Run 'Data_management/Create_faster_ovarian.py'. Set the DATA_DIR variable in the top of the file to the preprocessed folder that contains preprocessed_data.csv. 


## 2.2 LIDC dataset
Aside from Ovarian dataset, our study compares against the public LIDC IDRI lung nodule. This subsection will detail how to preprocess the LIDC dataset so that it is in the same format as the ovarian dataset. Make sure that you use GP_3 interpreter for these steps. 

- [**REQUIRED**]  download the LIDC dataset here: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254
You have to use their tool *NBIA Data Retriever" to download the dataset. Before you download, **make sure to select the checkbox at 'Classic Directory Name'** This makes sure that the format of the dataset will be correct.
For linux, download the deb and install using `sudo -S dpkg -r nbia-data-retriever-4.4.2.deb;sudo -S dpkg -i nbia-data-retriever-4.4.2.deb`Then it is added as a program, so you can launch it using search bar (if you get manifest error, download .tcia file for the LIDC dataset and double click the file. 

- [**REQUIRED**] We use the PyLIDC dataset for preprocessing. To set up PyLIDC, follow this guide. 
https://pylidc.github.io/install.html 

- [**REQUIRED**]  We convert the LIDC dataset to Nifti format so that it is in the same format and data structure as our ovarian data. This is done using the script (```/Data_management/Preprocess_LIDC_to_NIFTI.py```)

- [**REQUIRED**] Similarly to the LIDC dataset, run  `/ovacadx/scripts/preprocess_data.py` on the LIDC NIFTI directory to create a copy of standardized CT-scans,  Then run `Data_management/Create_faster_ovarian.py` to create a cropped copy of the dataset for faster data loading. Set DATA_DIR in the top of the file to the preprocessed folder that contains preprocessed_data.csv. 

After following the steps for both ovarian and lung nodule dataset, you should have 4 dataset folders:
1. Ovarian/NIFTI
2. Ovarian/NIFTI_FAST
3. LIDC/NIFTI
4. LIDC/NIFTI_FAST

where the \_FAST datasets are the datasets that contain only the crops of the tumors (i.e. these datasets are much smaller and thus faster for our deep learning applications). 


# 3. Extracting radiomic features
For any radiomics-based experiments, it is a good idea to open the project in the radiomics folder; open the project with '/ovacadx/' as the root directory. Almost all radiomics-based files can be found in `/ovacadx/scripts/Radiomics/`.
  
To run radiomic experiments, you first need to extract radiomic features. As extracting features takes some time, we extract the features once and save them on the disk. 

- Run "/Code/ovacadx/scripts/Radiomics/extract_features.py" . This will save a 3DRadiomicfeatures.json in the root directory of the dataset. This file will be read in the radiomic-based experiments. 

Generally for radiomics-based experiments, the order of running files looks something like this: 
1. Run /Code/ovacadx/scripts/Radiomics/train.py which trains 25-fold nested cross validation. It is important that for the radiomics experiments, you set the correct feature selection pipeline you want to use. For more details on this, see 3.1 in this readme.
2. Run test.py on the directory that contains the train results
3. Run /Code/ovacadx/scripts/Radiomics/Analysis/ens_no_logits2.py. This file ensembles the test-set results for each of the inner loops, and then applies bootstrapping to obtain emperical confidence intervals. The resulting metrics are stored in "results[MODEL_NAME]_compiled_summary.csv"
- I highly recommend to combine all of these steps for reproducilbility. For an example, check '/Code/ovacadx/scripts/Radiomics/Experiment_radiomics_SPIE/Experiment_SPIE.py'. This file first runs train.py, then test.py and then ens_no_logits2.py. Doing it like this ensures consistency in hyperparameters and makes it easier to reproduce your experiments. 

## 3.1 Setting the feature selection pipeline in radiomics experiments
The feature selection pipeline in train.py and test.py is defined by two command-line arguments: `--pipeline` and `--features`. The pipeline specifies the sequence of feature selection techniques to apply, while the features argument indicates how many features to retain at each step in the sequence.

### How the Pipeline Works:

1.  **Pipeline Steps (`--pipeline`)**: This argument is a comma-separated list of feature selection methods. Each method represents a different technique for selecting or transforming features from the dataset.
    
2.  **Number of Features (`--features`)**: This argument is a comma-separated list of integers that correspond to the number of features to keep at each stage in the pipeline. Each number in this list should match the respective feature selector in the pipeline.
    
3.  **Mapping Pipeline Steps to Feature Count**: The pipeline and features lists must have the same length. For each feature selection step, the corresponding number in the features list dictates how many features will be retained or selected. If the lengths of these two lists don’t match, an error is raised.
    

### Example of Setting a Feature Selection Pipeline:
For example if we run
`/Code/ovacadx/scripts/Radiomics/train.py --pipeline SELECT_PERCENTILE,PS,BLACK_BOX --features 10,40,0`
This will happen:

-  First, all radiomic features will be read from the json file.
-   Then, 10% of the features are selected using percentile ranking (univariate analysis).
-   Next, 40 features are chosen using the Pearson-Spearman correlation method. (For details, see the graduation report)
-   Finally, the black-box Random Forest-based feature elimination is applied without a predefined feature count (determined automatically).
- After this, the four classifiers (NN, RF, SVM, LR) are trained.

Coincidentally, the pipeline described above is the "default pipeline" that is used in our publication, adopted from Li et al. 

All options that are implemented for feature selection are detailed below:

- BLACK_BOX: Uses greedy feature elimination based on random-forest feature importance. The number of features you use does not matter, as this function automatically determines the optimal number of features to retain. Beware to use other feature selectors before black box, as the algorithm is very slow for large numbers of features!
- PCA: Principal component analyisis. The number of features is the number of prinipal components that are calculated. Please note that PCA can only be applied as the last step of the feature selection pipeline! (Other options not implemented due to time constraints) 
- SELECT_PERCENTILE: uses univariate analysis ( https://scikit-learn.org/1.5/modules/generated/sklearn.feature_selection.SelectPercentile.html) 
- PS: uses peason-spearman correlation, adopted from Li et al. and then modified for better performance Please check the graduation report for details on the modifications (iterative approach instead of fixed threshold).
- ANOVA: Analysis of variance 
- KS: Kolmogorov Smearnov feature selection, keeping N best the features with lowest interclass similarity.
- WD: Wasserstein distance feature selection, similar to KS but uses wasserstein distance. 
- KS_alpha: KS feature selection, but now with option alpha so that you can mix between interclass and intraclass features. When alpha = 0, features with optimal intraclass similarity dominate. When alpha = 1, features with optimal interclass similarity dominate. 

# 4. Deep Learning Instructions
For our deep learning experiments, we keep the order of train.py, test.py, ens_no_logits2.py. Note that we use the ens_no_logits2.py that is located in /ovacadx/scripts/Radiomics/Analysis/ens_no_logits2.py, so that both radiomics and deeplearning experiments have the same bootstrapping analysis applied. Again, it is good to use a single python file that runs all needed files at once. For example, see "/MIL/experiments/experiment_ovarian_baseline/Ovarian_baseline.py". 

If you experience dataloader crashes, consider changing '--num_workers 15'. (Reduce for less speed but more stability) 

# 5. How to reproduce the results from the SPIE publication? 
## Figure 1. 
Run `/MIL/experiments/experiment_show_tumors/show_tumors.py` 

## Table 1.
Run `ovacadx/scripts/Radiomics/Experiment_statsistics/statistics.py` The LIDC dataset row was added manually. 
Note that the script first puts out the latex table, and AFTER that it puts out another table which is patient-wise distribution. The latter is not used in the SPIE publication.

## Figure 2.
This figure was made in Adobe Illustrator.  

## Figure 3.
This figure was made in Adobe Illustrator.  

## Figure 4.
Run `/ovacadx/scripts/Radiomics/Experiment_plot_ks_SPIE/Experiment_plot_ks.py`

## Table 2.
AFTER training, testing and ensembling all model scores (see section 6):
Run `ovacadx/scripts/Radiomics/Experiment_score_table_2_SPIE/score_table.py ` Make sure to specify "csv_paths", "model_names", "data_labels" and "output_csv_path" in the bottom of the code of the python file. The code will produce the Latex table. 


## Figure 5
AFTER training, testing and ensembling all model scores (see section 6):
Run `ovacadx/scripts/Radiomics/Analysis/Half_subset_auc.py`
Note that the first time you run this, you must uncomment the code that evaluates all the half-group AUCs. That section of code will save the scores and polarities to a file 'half_group_side.pkl' and 'half_group_score.pkl', which can be used in subsequent analysis. Hence after running that section of code once, it can be commented again for convenience :). 


# 6 Reproduction of our model trainings (for tables)
In our SPIE publication, we present 5 models. Instructions on how to train each model are below: 

General note: If you want to train the models, look through each file for "#os.system(command)", remove the # so that the line will run. We have commented these lines out for our convenience (when re-running an analysis script for example), so that the model does not re-train when we are debugging evaluation for example. So, just make sure that os.system is uncommented everywhere if you wish to retrain. 

1. Run `ovacadx/scripts/Radiomics/Experiment_radiomics_SPIE/Experiment_SPIE.py`. This file will generate all results for the four different classifiers (SVM, RF, NN, LR) for both ovarian and lund nodule datasets. Make sure that you specify the output directories in 'experiment_folders' and the dataset directories in "data_dirs" at the top of the python file. 
Use the default pipeline, i.e. use pipeline = ['SELECT_PERCENTILE','PS','BLACK_BOX'], features = ['10','40','0']. Note that in the SPIE publication, only the result using the NN classifier was published. This is because the difference between classifiers is small and we figured that it is better to only have a single radiomics result as a baseline. 

2. Run `/MIL/experiments/experiment_ovarian_baseline/Ovarian_baseline.py`. Make sure to configure 'experiment_folder' and 'data_dir' in the header of the python file. "my_type" should be "OVARY"
3. Run `/MIL/experiments/experiment_lung_baseline/exp_lung_baseline.py` Make sure to configure 'experiment_folder' and 'data_dir' in the header of the python file. "my_type" should be "LUNG"
4. Run `MIL/experiments/experiment_3DCNN_ovarian/exp_3DCNN_ovarian_baseline.py` Make sure to configure 'experiment_folder' and 'data_dir' in the header of the python file. "my_type" should be "OVARY" 
5. Run `MIL/experiments/experiment_3DCNN_LUNG/experiment_3DCNN_LUNG.py` Make sure to configure 'experiment_folder' and 'data_dir' in the header of the python file. "my_type" should be "LUNG" 

Now, we must also train the deep-learning models with the rotation augmentation enabled. 

6. Run `/MIL/experiments/experiment_ov_rotate/Ov_rotate.py`. Make sure to configure 'experiment_folder' and 'data_dir' in the header of the python file. "my_type" should be "OVARY"
7. Run `MIL/experiments/experiment_lung_rotate/lung_rotate.py` Make sure to configure 'experiment_folder' and 'data_dir' in the header of the python file. "my_type" should be "LUNG"
8. Run `MIL/experiments/experiment_3DCNN_OvarianRotate/3dcnn_ovarian_rotate.py` Make sure to configure 'experiment_folder' and 'data_dir' in the header of the python file. "my_type" should be "OVARY" 
9. Run `MIL/experiments/experiment_3DCNN_LIDCRotate/3dcnn_lung_rotate.py` Make sure to configure 'experiment_folder' and 'data_dir' in the header of the python file. "my_type" should be "LUNG" 


# 7 Footnote on AUC_median
Our publication states: "For the full list of AUCmedian of all radiomic features, see https://github.com/EloySchultz/OvaCADx_SPIE2025"
The full list of AUC_median can be found here: `ovacadx/scripts/Radiomics/Analysis/Optimal_half_AUCs_per_feature.csv`. This list was generated using `ovacadx/scripts/Radiomics/Analysis/Half_subset_auc.py` (by uncommenting the section that evaluates all half-group AUCs). 


 ## Disclaimer
 The code in this repo was copied from a larger, internal repository that also includes files that were used during development. I hope that I copied all relevant files, but if some are missing, please raise an issue in this repo. 
 


