import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent

# important subdirectories
data_dir = repo_root / 'data'
results_dir = repo_root / 'results'

# location of saved expression data
# mad_data = data_dir / 'tcga_mad_genes.tsv'
pancan_data = data_dir / 'pancancer_data.pkl'
rnaseq_data = data_dir / 'tcga_expression_matrix_processed.tsv.gz'
sample_counts = data_dir / 'tcga_sample_counts.tsv'
sample_info = data_dir / 'tcga_sample_identifiers.tsv'

# location of test data
test_data_dir = repo_root / 'tests' / 'data'
test_expression = test_data_dir / 'expression_subsampled.tsv.gz'

# parameters for classification using raw gene expression
num_features_raw = 8000

# hyperparameters for classification experiments
filter_prop = 0.05
filter_count = 15
train_folds = 3
max_iter = 200
alphas = [0.01, 0.1, 1, 10, 100]
l1_ratios = [0.0, 0.25, 0.5, 0.75, 0.9]
learning_rates = [0.001, 0.0001, 1e-5]

# default seed for random number generator
default_seed = 42
