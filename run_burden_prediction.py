"""
Script to run mutation burden prediction experiments
"""
import os
import sys
import subprocess

import pandas as pd
from tqdm import tqdm

import pancancer_utilities.config as cfg

EXP_SCRIPT = os.path.join(cfg.repo_root,
                          'pancancer_utilities',
                          'scripts',
                          'predict_mutation_burden.py')
MAD_GENES = 8000

def get_all_cancer_types():
    counts_df = pd.read_csv(os.path.join(cfg.data_dir, 'tcga_sample_counts.tsv'),
                            sep='\t')
    return counts_df.cancertype.values

def run_single_experiment(cancer_type, use_pancancer, shuffle_labels,
                          verbose=False):
    args = [
        'python',
        EXP_SCRIPT,
        '--holdout_cancer_type', cancer_type,
        '--subset_mad_genes', str(MAD_GENES)
    ]
    if use_pancancer:
        args.append('--use_pancancer')
    if shuffle_labels:
        args.append('--shuffle_labels')

    if verbose:
        print('Running: {}'.format(' '.join(args)))
    subprocess.call(args)

if __name__ == '__main__':

    cancer_types = get_all_cancer_types()
    # do single cancer first, much faster
    for cancer_type in tqdm(cancer_types):
       run_single_experiment(cancer_type, False, False, verbose=True)
       run_single_experiment(cancer_type, False, True, verbose=True)
    # then pancancer, this will be slower
    for cancer_type in tqdm(cancer_types):
       run_single_experiment(cancer_type, True, False, verbose=True)
       run_single_experiment(cancer_type, True, True, verbose=True)

