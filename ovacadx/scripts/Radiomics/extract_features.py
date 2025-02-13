#Extract radiomics.
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
import os
import argparse
from tqdm.contrib.concurrent import process_map
import radiomics as rad
import logging
import json
import time


def get_args_parser():
    parser = argparse.ArgumentParser(description='Extract radiomic features from dataset')
    parser.add_argument('--data_path', type=str,
                        default='/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/')  # default='/path/to/output/folder', help='Path to output folder')
    parser.add_argument('--mp', type=bool, default = True, help="Enable or disable multiprocessing (faster)")
    parser.add_argument('--debug', type=bool, default = False, help="Enable debugging")
    return parser
def extract_radiomics(s_dict):
    try:
        data_path = s_dict['data_path']
        extractor = s_dict['extractor']
        row = s_dict['row']
        IP = os.path.join(data_path, row['image_path'])
        AP = os.path.join(data_path, row['annot_path'])
        featureVector = extractor.execute(IP, AP)
        featureVector['original_class'] = row['label']
        featureVector['is_internal_testset_tiny'] = row['is_internal_testset_tiny']
        featureVector['is_internal_testset'] = row['is_internal_testset']
        featureVector['is_external_testset'] = row['is_external_testset']
        ID = row['tumor_id']
        # Remove diagnostics
        feature_names = [k for k in featureVector.keys() if not k.startswith("diagnostics")]
        featureList = []
        for fname in feature_names:
            featureList.append(featureVector[fname])
        featureList = list(np.array(featureList))  # Removes array()'s'
        return featureList, ID, feature_names
    except Exception as e:
        print(f"Error extracting features for image: {IP}")
        print(f"Error message: {str(e)}")
        return None  # You can handle the error as needed
def main(args):
    df = pd.read_csv(os.path.join(args.data_path,"preprocessed_data.csv"))
    if not 'is_internal_testset_tiny' in df.keys():
        raise ValueError("Tiny test set missing from preprocessed_data.csv. Please copy dataset.csv from scripts/Radiomics/Datasets_csv and run preprocess_data.py")

    featureClasses = rad.getFeatureClasses()

    params = os.path.join("Params3D.yaml")

    extractor = rad.featureextractor.RadiomicsFeatureExtractor(params)
    print('Will extract the radiomics from the following image types:')
    for imageType in extractor.enabledImagetypes.keys():
        print('\t' + imageType)

    # Disable all classes
    extractor.disableAllFeatures()

    enableFeatureClasses = featureClasses
    #Disable 2D radiomic features
    if "shape2D" in enableFeatureClasses.keys():
        enableFeatureClasses.pop("shape2D")

    for fc in enableFeatureClasses:
        # Enable all features in firstorder
        extractor.enableFeatureClassByName(fc)
    #Count the features
    result = []
    # for cls in enableFeatureClasses.values():
    #     result.extend(list(cls.getFeatureNames().values())) #Depricated will be true, non-depricated will be false.
    # num_features = sum([not item for item in result])


    logger = logging.getLogger("radiomics.glcm")
    logger.setLevel(logging.ERROR)  # Disable glcm symmetry error

    #radiomicFeatures3D = np.zeros((1, num_features+4))  # 107 features + class + is_internal_testset_tiny + is_internal_testset + is_external_test_set
    print("ETA is inaccurate, but extracting radiomics is slow. Wait patiently. Enabling MP and setting an appropriate number of workers helps.")
    time.sleep(5)
    rows = [{'row':row,'data_path':args.data_path,'extractor':extractor} for index, row in df.iterrows()]
    if args.debug:
        n=50
        rows = rows[:n] #For testing purposes
        print("Warning, Debug enabled!")
        print("Only extracting features from " + str(n) + " number of scans")
    if args.mp:
        print("Multiprocessing enabled!")
        results = process_map(extract_radiomics, rows, max_workers=10, chunksize=1)
    else:
        print("Multiprocessing disabled! This will be slow...")
        results=[]
        for row in tqdm(rows, desc="Processing", unit="scan"):
            results.append(extract_radiomics(row))
    IDs = []  # Saving filenames here.
    radiomicFeatures3D = []
    for result in results:
        if isinstance(result, type(None)):
            continue
        if len(radiomicFeatures3D) == 0:
            radiomicFeatures3D=np.zeros((1, len(np.array(result[0]))))
        # radiomicFeatures3D_dict[result[1]]=result[0]
        IDs.append(result[1])
        radiomicFeatures3D = np.vstack((radiomicFeatures3D, np.array(result[0])))
        feature_names = result[2]

    # remove 0 row
    radiomicFeatures3D = radiomicFeatures3D[1:]
    combined_data = {'arr': radiomicFeatures3D.tolist(), 'ids': IDs, 'names': feature_names}

    # Define the file path
    file_path = os.path.join(args.data_path, "3DRadiomicfeatures.json")

    with open(file_path, 'w') as file:
        json.dump(combined_data, file)

    print("Data saved to:", file_path)

    return
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
