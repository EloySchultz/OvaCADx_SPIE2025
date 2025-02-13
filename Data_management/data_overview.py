import argparse
import pandas as pd
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser(description='MIL CNN train script')
    parser.add_argument('--data_path', type=str, default="", metavar='N', help="Path to preprocessed_data.csv")
    return parser
def main(args):
    df = pd.read_csv(args.data_path)
    lung=False
    if "LIDC" in df['tumor_id']:
        lung=True

    if lung==False:
        # Step 1: Extract prefixes
        df['prefix'] = df['tumor_id'].apply(lambda x: x.split('_')[0])

        # Step 2: Define rows and columns
        rows = ["Benign", "Malignant"]
        centers = df['prefix'].unique()

        # Initialize the output DataFrame with the desired rows and columns
        out_df = pd.DataFrame(index=rows, columns=centers)

        # Fill the DataFrame with 100s
        for center in centers:
            for cls, cls_name in zip([0, 1], rows):
                out_df.loc[cls_name, center] = len(df[(df['label']==cls) & (df['prefix']==center)])
        for center in centers:
            out_df.loc['Total',center] = out_df.loc['Malignant',center] + out_df.loc['Benign',center]
        for center in centers:
            out_df.loc['Outliers', center] = len(df[(df['outlier']==1.0) & (df['prefix']==center)])
        print(out_df)
    pass

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
