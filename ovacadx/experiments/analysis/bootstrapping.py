import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def get_args_parser():
    parser = argparse.ArgumentParser(description='Bootstrapping of nested crossvalidation results')
    parser.add_argument('--data_path', type=str, default='/path/to/results/csv')
    parser.add_argument('--num_bootstraps', type=int, default=1000, help='Number of bootstraps')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
    
    return parser


def main(args):
    # Load results
    results = pd.read_csv(args.data_path)

    # Set seed
    np.random.seed(args.seed)
    
    aucs_internal = []
    aucs_external = []
    for _ in range(args.num_bootstraps):
        labels_internal = []
        logits_internal = []
        labels_external = []
        logits_external = []    
        for _, row in results.iterrows():
            label = 1 if 'M' in row['tumor_id'].split('_')[1] else 0
            labels_external.append(label) if 'AVL' in row['tumor_id'] else labels_internal.append(label)
            # random pick one of the columns that is not 'tumor_id' and does not contain nan
            logit = row[1:].dropna().sample().item()
            logits_external.append(logit) if 'AVL' in row['tumor_id'] else logits_internal.append(logit)
        aucs_internal.append(roc_auc_score(labels_internal, logits_internal))
        aucs_external.append(roc_auc_score(labels_external, logits_external))
    
    print("--- Internal AUC ---")
    print(f"Mean AUC: {sum(aucs_internal) / len(aucs_internal)}")
    print(f"95% CI: [{sorted(aucs_internal)[int(0.025 * args.num_bootstraps)]}, {sorted(aucs_internal)[int(0.975 * args.num_bootstraps)]}]")
    print("--- External AUC ---")
    print(f"Mean AUC: {sum(aucs_external) / len(aucs_external)}")
    print(f"95% CI: [{sorted(aucs_external)[int(0.025 * args.num_bootstraps)]}, {sorted(aucs_external)[int(0.975 * args.num_bootstraps)]}]")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
