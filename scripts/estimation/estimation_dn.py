"""
    Returns a dictionary of predicted neural network models. Keys are sample keys.
"""
# Import Modules
import sys
from joblib import Parallel, delayed
sys.path.append('./scripts/estimation')
import network_functions as nf
import pandas as pd
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

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
activation_function = 'relu'  # Hidden layer transformation function
loss_fn = 'mean_squared_error'  # Loss function.
learning_rate = 1e-3  # Learning rate.
epsilon = 1e-8  # Adam epsilon
mini_batch_training_batch_size = 128  # Batch size
mini_batch_training_epoch_limit = 50  # Max Number of iterations
mini_batch_tol = 1e-8  # Tolerance to stop iterations
n_hidden_node_search_distance = 5  # Half-range for number of node search

# Import Data
full_data = pd.read_csv(data_path, index_col=data_index_column_name_identifier)  # Load data
idx_bootstrap = np.load(idx_bootstrap_data_path).item()  # Load indices for each bootstrap sample

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


def cross_validation(sample_key):  # Define optimization algorithm with cross-validation
    from tensorflow import keras  # Import keras for each thread
    keras.backend.set_session(tf.Session(config=config))  # Set backend with config options
    optimizer = keras.optimizers.Adam(lr=learning_rate, epsilon=epsilon)  # Set optimizer
    # Pick training and cross-validation data for this particular bootstrap
    idx_training = idx_bootstrap[sample_key]['training_sample']  # Retrieve training set
    idx_cv = idx_bootstrap[sample_key]['cv_sample']  # Retrieve cross-validation set
    x_train, y_train = nf.prepare_data(full_data, p_ident, e_ident, b_ident,
                                       d_ident=d_ident, idx=idx_training)  # Retrieve input and output for training
    x_cv, y_cv = nf.prepare_data(full_data, p_ident, e_ident, b_ident,
                                 d_ident=d_ident, idx=idx_cv)  # Retrieve input and output for cross-validation

    # Estimation
    def estimation(n_node):  # Define optimization routine
        file_path = "./output/temp/dn/cross_validation/sample_{}_node_{}.h5".format(sample_key, n_node)  # Save path
        dn_model = nf.build_model(n_node, n_goods, optimizer, loss_fn, activation_fn=activation_function)  # Set model
        callbacks = [keras.callbacks.EarlyStopping('loss', min_delta=mini_batch_tol)]  # Define callbacks
        history = dn_model.fit(x=x_train, y=y_train, batch_size=mini_batch_training_batch_size,  #Optimization
                               epochs=mini_batch_training_epoch_limit, callbacks=callbacks, verbose=0,
                               validation_data=(x_cv, y_cv))
        dn_model.save(file_path)  # Save optimization output
        return history.history  # Return output
    cv_results = {'Number of Nodes': [], 'Loss History': []}  # Pre-allocate result dictionary
    for node in n_hidden_node_search_set:  # Loop over potential number of nodes
        cv_results['Number of Nodes'].append(node)
        cv_results['Loss History'].append(estimation(node))
    keras.backend.clear_session()  # Reset session
    return cv_results


output = Parallel(n_jobs=n_cores_to_joblib, verbose=10)(delayed(cross_validation)(sample_key)
                                                        for sample_key in range(n_bootstrap))
