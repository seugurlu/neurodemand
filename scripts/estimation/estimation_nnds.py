"""
    Returns a dictionary of predicted neural network models. Keys are sample keys.
"""
# Import Modules
import sys
from joblib import Parallel, delayed
sys.path.append('./scripts/estimation')
import network_functions as nf
import function_nnds as nnds
import pandas as pd
import numpy as np
import tensorflow as tf


# Set system related hyper-parameters
n_cores_to_tf = 1
n_cores_to_joblib = 6

# Set tensorflow related options
config = tf.ConfigProto(
    intra_op_parallelism_threads=n_cores_to_tf,
    inter_op_parallelism_threads=n_cores_to_tf)

# Set Data Related Hyper-parameters
starting_sample = 0
n_bootstrap = 200  # Number of bootstrapped samples.
p_ident = 'pt_'  # String identifier for price columns.
e_ident = 'total_expenditure'  # String identifier for total expenditure column.
b_ident = 'wt_'  # String identifier for budget share columns.
d_ident = None  # String identifier for demographic variables columns. Set to 'd_' for estimation with demographics.
data_path = "./output/sample_adjustments/edited_data.csv"  # Path to the training data.
data_index_column_name_identifier = 'index'  # Column name that holds data point indices in input files.
idx_bootstrap_data_path = "./output/sample_adjustments/idx_bootstrap.npy"  # Path to bootstrap indices.

# Neural Network and Optimization Hyper-parameters
activation_fn = 'relu'  # Hidden layer transformation function
loss_fn = 'mean_squared_error'  # Loss function.
learning_rate = 1e-3  # Learning rate.
epsilon = 1e-8  # Adam epsilon
mini_batch_training_batch_size = 128  # Batch size
mini_batch_training_epoch_limit = 50  # Max Number of iterations
mini_batch_tol = 1e-8  # Tolerance to stop iterations
n_hidden_node_search_distance = 5  # Half-range for number of node search

# Import Data
full_data = pd.read_csv(data_path, index_col=data_index_column_name_identifier)  # Load data
idx_bootstrap = np.load(idx_bootstrap_data_path, allow_pickle=True).item()  # Load indices for each bootstrap sample

# Extract some data-related hyper-parameters
n_goods = full_data.columns.str.startswith(b_ident).sum()  # Retrieve number of goods
if d_ident is None:
	n_demographics = 0
else:
	n_demographics = full_data.columns.str.startswith(d_ident).sum()  # Retrieve number of demographics

# Set data reliant neural network hyper-parameters
n_hidden_node_search_midpoint = int(np.sqrt(
        (n_goods + n_demographics + 1) * n_goods
    ))  # Set midpoint for node search as geometric mean of number of nodes in input and output layers
n_hidden_node_search_set = nf.generate_hidden_search_set(n_hidden_node_search_midpoint,
                                                         n_hidden_node_search_distance)  # Set hidden node search set

# Cross-validation


def nnds_estimation(sample_key):  # Define optimization algorithm with cross-validation
	from tensorflow import keras  # Import keras for each thread
	nnds_output = nnds.cross_validation(sample_key, idx_bootstrap, full_data, n_hidden_node_search_set,
										p_ident, e_ident, b_ident, d_ident,
										loss_fn, activation_fn,
										mini_batch_training_batch_size, mini_batch_tol, mini_batch_training_epoch_limit,
										learning_rate, epsilon, config)
	keras.backend.clear_session()  # Reset session
	return nnds_output


output = Parallel(n_jobs=n_cores_to_joblib, verbose=10)(delayed(nnds_estimation)(sample_key)
                                                        for sample_key in range(n_bootstrap))
