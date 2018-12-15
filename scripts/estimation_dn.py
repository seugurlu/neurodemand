"""
    Returns a dictionary of predicted neural network models. Keys are sample keys.
"""
# Import Modules
import time
import sys
sys.path.append('./scripts')
import network_functions as nf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Set system related hyper-parameters
n_cores_to_tf = 7
# Set tensorflow related options
config = tf.ConfigProto(
    intra_op_parallelism_threads=n_cores_to_tf,
    inter_op_parallelism_threads=n_cores_to_tf)
keras.backend.set_session(tf.Session(config=config))


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
epsilon = 1e-4
mini_batch_training_batch_size = 128
mini_batch_training_epoch_limit = 50
mini_batch_tol = 1e-8
n_hidden_node_search_distance = 5
optimizer = keras.optimizers.Adam(lr=learning_rate, epsilon=epsilon)

# Import Data
full_data = pd.read_csv(data_path, index_col=data_index_column_name_identifier)
idx_bootstrap = np.load(idx_bootstrap_data_path).item()


# Extract some data-related hyper-parameters
n_goods = full_data.columns.str.startswith(b_ident).sum()
try:
    n_demographics = full_data.columns.str.startswith(d_ident).sum()
except AttributeError:  # If d_ident is None.
    n_demographics = 0

# Set data reliant neural network hyper-parameters
n_hidden_node_search_midpoint = int(np.sqrt(
        (n_goods + n_demographics + 1) * n_goods
    ))
n_hidden_node_search_set = nf.generate_hidden_search_set(n_hidden_node_search_midpoint, n_hidden_node_search_distance)

# Cross-validation


def cross_validation(sample_key):
    start_time = time.time()
    # Pick training and cross-validation data for this particular bootstrap
    print("Cross Validation starts with bootstrap sample {}".format(sample_key))
    idx_training = idx_bootstrap[sample_key]['training_sample']
    idx_cv = idx_bootstrap[sample_key]['cv_sample']
    x_train, y_train = nf.prepare_data(full_data, p_ident, e_ident, b_ident, d_ident=d_ident, idx=idx_training)
    x_cv, y_cv = nf.prepare_data(full_data, p_ident, e_ident, b_ident, d_ident=d_ident, idx=idx_cv)

    # Estimation
    def estimation(n_node):
        file_path = "./output/temp/dn/cross_validation/sample_{}_node_{}.h5".format(sample_key, n_node)
        dn_model = nf.build_model(n_node, n_goods, optimizer, loss_fn, activation_fn=activation_function)
        callbacks = [keras.callbacks.EarlyStopping('loss', min_delta=mini_batch_tol)]
        history = dn_model.fit(x=x_train, y=y_train, batch_size=mini_batch_training_batch_size,
                               epochs=mini_batch_training_epoch_limit, callbacks=callbacks, verbose=0,
                               validation_data=(x_cv, y_cv))
        dn_model.save(file_path)
        return history.history
    cv_results = {'Number of Nodes': [], 'Loss History': []}
    for node in n_hidden_node_search_set:
        cv_results['Number of Nodes'].append(node)
        cv_results['Loss History'].append(estimation(node))
    end_time = time.time()
    elapsed = (end_time - start_time)/60
    print("Predicted time remaining: {} minutes".format((n_bootstrap-sample_key-1)*elapsed))
    return cv_results

dn_models = {}
for i in range(starting_sample, n_bootstrap):
    dn_models[i] = cross_validation(i)
