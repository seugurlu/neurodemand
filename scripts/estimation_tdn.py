
"""
    Returns a dictionary of predicted neural network models. Keys are sample keys.
"""
# Import Modules
from joblib import Parallel, delayed
import sys
sys.path.append('C:/Users/seugurlu/Documents/Bitbucket Projects/neurodemand/scripts')
from helper_functions import *
import torch
import torch.nn as nn
import pandas as pd
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
torch.set_num_threads(6)
#number_cores = 1
device = torch.device("cpu")  # Device to do optimization with.
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

#Without CV
nn_model =  set_model(n_input_layer, 10, n_output_layer, activation_function, device)
optimizer = th.optim.Adam(nn_model.parameters(), learning_rate)
sample_key = 0
symmetry = True
negativity = True
theory_sample = None
nn_model, loss_matrix, cv_loss_matrix = nn_estimation(0, full_data, idx_bootstrap, p_ident, e_ident, b_ident,
                  nn_model, optimizer, loss_fn, symmetry=True, negativity=True,
                  d_ident=None, theory_sample=None, max_tol=1e-8,
                  max_iter=1000, batch_size=256, batch_size_factor=2, device='cpu', print_output=True, follow_cv=False)
# CV optimization
