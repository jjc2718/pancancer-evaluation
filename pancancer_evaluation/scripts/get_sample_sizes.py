"""
Script to get sample size information for gene/cancer type combinations

"""
import tqdm
import numpy as np
import pandas as pd

import pancancer_evaluation.config as cfg
from pancancer_evaluation.data_models.tcga_data_model import TCGADataModel
import pancancer_evaluation.utilities.data_utilities as du

def get_num_samples_for_gene(identifier, classification, tcga_data):
    # process data for given gene
    tcga_data.process_train_data_for_gene(gene, classification, None)
    assert tcga_data.X_train_raw_df.shape[0] == tcga_data.y_train_df.shape[0]
    return tcga_data.y_train_df.shape[0]


if __name__ == '__main__':

    tcga_data = TCGADataModel(verbose=True)

    # run for all genes in Vogelstein data set
    genes_df = du.load_vogelstein()
    all_genes = genes_df.gene.values

    sample_sizes = []

    for gene in tqdm.tqdm(all_genes):

        try:
            classification = du.get_classification(gene, genes_df)
            # TODO: num samples for each cancer type?
            num_samples = get_num_samples_for_gene(gene, classification, tcga_data)
            sample_sizes.append([gene, num_samples])
        except KeyError:
            # can happen for genes that aren't in mutation data, if so skip
            continue

    sample_size_df = pd.DataFrame(sample_sizes, columns=['gene', 'num_samples'])
    sample_size_df.to_csv('test.tsv', sep='\t')

