"""
    Returns a dictionary of predicted neural network models. Keys are sample keys.
"""
# Import Modules
import pandas as pd
import numpy as np
import tensorflow as tf
from sys import path
sys.path.append("./scripts")
import nn_functions as nf
import useful_functions as uf
import datetime

"""Set Data Related Hyper-parameters"""
data_path = "./output/sample_adjustments/edited_data.csv"  # Path to the training data.
data_index_column_name_identifier = 'index'  # Column name that holds data point indices in input files.
idx_bootstrap_data_path = "./output/sample_adjustments/idx_bootstrap.npy"  # Path to bootstrap indices.
p_ident = 'pt_'  # String identifier for price columns.
e_ident = 'total_expenditure'  # String identifier for total expenditure column.
b_ident = 'wt_'  # String identifier for budget share columns.
d_ident = None  # String identifier for demographic variables columns. Set to 'd_' for estimation with demographics.

"""Import Data"""
full_data = pd.read_csv(data_path, index_col=data_index_column_name_identifier)
idx_bootstrap = np.load(idx_bootstrap_data_path).item()

"""Set Network Related Hyper-parameters"""
model_dir = "./output/temp/logs/" + datetime.datetime.now().strftime("%y%m%d%H%M%S")  # Set to None for temp folder.
activation_function = tf.nn.relu  # Hidden layer transformation function
hidden_units = [12]

"""Set Optimization Related Hyper-parameters"""
n_cores = 6
learning_rate = 1e-3
epsilon = 1e-4
batch_size = 128
max_iter = 50  # Maximum number of iterations
max_tol = 1e-8  # Allowed convergence tolerance
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)  # Define the optimizer
validation_steps = 5  # Epochs to evaluate with cross-validation set. Set to None if no evaluation until convergence.
print_steps = 5  # Set to None if in silent mode

"""Set tensorflow related options"""
session_config = tf.ConfigProto(
    intra_op_parallelism_threads=n_cores,
    inter_op_parallelism_threads=n_cores)
tf.estimator.RunConfig(
    model_dir=model_dir,
    save_summary_steps=int(n_training/batch_size),
    save_checkpoints_secs=None,
    save_checkpoints_steps=None,
    session_config=session_config,
)

"""Retrieve training and cross-validation samples for a specific bootstrap sample and declare input types."""
# Select bootstrap sample and prepare data input functions
