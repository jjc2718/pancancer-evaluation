"""
Script to get sample size information for gene/cancer type combinations

"""
import tqdm
import numpy as np
import pandas as pd

import pancancer_evaluation.config as cfg
from pancancer_evaluation.data_models.tcga_data_model import TCGADataModel
import pancancer_evaluation.utilities.data_utilities as du

def count_samples_for_gene(identifier, classification, tcga_data):
    tcga_data.process_train_data_for_gene(gene, classification, None)
    assert tcga_data.X_train_raw_df.shape[0] == tcga_data.y_train_df.shape[0]
    id_counts_df = (
        tcga_data.y_train_df.groupby('DISEASE')
                            .count()
                            .reset_index()
                            .rename({'status': 'count', 'DISEASE': 'cancer_type'},
                                    axis='columns')
                            .assign(identifier=lambda x: gene + '_' + x.cancer_type)
    )
    id_counts_df.drop(id_counts_df.columns.difference(['identifier', 'count']),
                      axis='columns', inplace=True)
    return id_counts_df


if __name__ == '__main__':

    tcga_data = TCGADataModel(verbose=True)

    # run for all genes in Vogelstein data set
    genes_df = du.load_vogelstein()
    all_genes = genes_df.gene.values

    id_counts_df = pd.DataFrame()

    for gene in tqdm.tqdm(all_genes):

        try:
            classification = du.get_classification(gene, genes_df)
            id_counts_df = pd.concat((
                id_counts_df,
                count_samples_for_gene(gene, classification, tcga_data)
            ))
        except KeyError:
            # can happen for genes that aren't in mutation data, if so skip
            print('Skipping gene {}, not in mutation data'.format(gene))
            continue

    id_counts_df.to_csv(cfg.id_sample_counts, sep='\t', index=False)

