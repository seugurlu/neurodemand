
"""
    Returns a dictionary of predicted neural network models. Keys are sample keys.
"""
# Import Modules
from joblib import Parallel, delayed
import sys
sys.path.append('C:/Users/seugurlu/Documents/Bitbucket Projects/neurodemand/scripts')
#from helper_functions import *
import torch as th
import pandas as pd
import torch.nn as nn
import numpy as np


# Set Data Related Hyper-parameters
n_bootstrap = 200  # Number of bootstrapped samples.
p_ident = 'pt_'  # String identifier for price columns.
e_ident = 'total_expenditure'  # String identifier for total expenditure column.
d_ident = None  # String identifier for demographic variables columns. Set to 'd_' for estimation with demographics.
b_ident = 'wt_'  # String identifier for budget share columns.
data_path = "./output/sample_adjustments/edited_data.csv"  # Path to the training data.
data_index_column_name_identifier = 'index'  # Column name that holds data point indices in input files.
idx_bootstrap_data_path = "./output/sample_adjustments/idx_bootstrap.npy"  # Path to bootstrap indices.
n_goods = 10
n_demographics = 0


# Neural Network and Optimization Hyper-parameters
th.set_num_threads(1)
number_cores = 7
device = th.device("cpu")  # Device to do optimization with.
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Uncomment for GPU use if available.
activation_function = nn.ReLU()  # Hidden layer transformation function.
loss_fn = nn.MSELoss()  # Loss function.
learning_rate = 1e-4  # Learning rate.
max_iter = 200     # Max number of iterations before convergence.
batch_size = 32  # Starting batch size.
batch_size_factor = 2  # Factor to increase batch_size if convergence is not achieved with the existing batch_size.
max_tol = 1e-8  # Tolerance level for convergence.
print_output = False  # If optimization information is printed.
follow_cv = False  # If cross-validation loss is followed during optimization.
n_hidden_min = 2
n_hidden_max = 2*n_goods
n_input_layer = n_goods + 1 + n_demographics
n_output_layer = n_goods
cv_check_start = 10

# Import Data
full_data = pd.read_csv(data_path, index_col=data_index_column_name_identifier)
idx_bootstrap = np.load(idx_bootstrap_data_path).item()

# CV optimization


def estimator(sample_key):
    outputs = nn_estimation_with_cv(sample_key, n_input_layer, n_output_layer, activation_function,
                                    n_hidden_min, n_hidden_max, loss_fn, idx_bootstrap, full_data, p_ident, e_ident,
                                    b_ident, d_ident, max_tol, max_iter, batch_size, batch_size_factor, device,
                                    print_output, follow_cv, learning_rate, cv_check_start)
    return outputs


output = Parallel(n_jobs=number_cores, verbose=10)(delayed(estimator)(sample_key)
                                                   for sample_key in range(n_bootstrap))
dnn_models = dict(list(zip(np.arange(n_bootstrap), output)))
np.save("./output/estimation/dnn_estimates.npy", dnn_models)
