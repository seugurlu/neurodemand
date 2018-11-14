"""
    Returns a dictionary of predicted neural network models. Keys are sample keys.
"""
# Import Modules
from joblib import Parallel, delayed
import sys
sys.path.append("./scripts")
from network_functions import *
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Set Data Related Hyper-parameters
n_bootstrap = 200  # Number of bootstrapped samples.
p_ident = 'pt_'  # String identifier for price columns.
e_ident = 'total_expenditure'  # String identifier for total expenditure column.
b_ident = 'wt_'  # String identifier for budget share columns.
d_ident = None  # String identifier for demographic variables columns. Set to 'd_' for estimation with demographics.
data_path = "./output/sample_adjustments/edited_data.csv"  # Path to the training data.
data_index_column_name_identifier = 'index'  # Column name that holds data point indices in input files.
idx_bootstrap_data_path = "./output/sample_adjustments/idx_bootstrap.npy"  # Path to bootstrap indices.

# Import Data
full_data = pd.read_csv(data_path, index_col=data_index_column_name_identifier)
idx_bootstrap = np.load(idx_bootstrap_data_path).item()


# Extract some data-related hyper-parameters
n_goods = full_data.columns.str.startswith(b_ident).sum()
try:
    n_demographics = full_data.columns.str.startswith(d_ident).sum()
except AttributeError:  # If d_ident is None.
    n_demographics = 0

# Neural Network and Optimization Hyper-parameters
# #Set number of threads for tensorflow HERE
number_cores = 7
activation_function = 'relu'  # Hidden layer transformation function
loss_fn = 'mse'  # Loss function.
learning_rate = 1e-3  # Learning rate.
epsilon = 1e-4
epoch = 50     # Max number of iterations before convergence.
batch_size = 256  # Starting batch size.
metric = ['mse']  # To evaluate cv sample during training
# batch_size_factor = 2  # Factor to increase batch_size if convergence is not achieved with the existing batch_size.
small_batch_tol = 1e-4
large_batch_tol = 1e-6
batch_tol = 1e-8  # Tolerance level for convergence.
# print_output = False  # If optimization information is printed.
# follow_cv = False  # If cross-validation loss is followed during optimization.
# n_hidden_min = 2
# n_hidden_max = 2*n_goods
# n_input_layer = n_goods + 1 + n_demographics
# n_output_layer = n_goods
# cv_check_start = 10
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Optimizer


# CV optimization

#Select sample
sample_key = 0
idx_training = idx_bootstrap[sample_key]['training_sample']
idx_cv = idx_bootstrap[sample_key]['cv_sample']
n_training_obs = idx_training.shape[0]
cv = pd_to_tfdata(full_data, p_ident, e_ident, b_ident, d_ident=d_ident, idx=idx_cv)

#Model
number_node = 12
dn_model = build_model(number_node, n_goods, optimizer, loss_fn, activation_fn='relu', metrics=metric)


# Small batch training for fast convergence to target area
batch_size = 32
steps_per_epoch = int(n_training_obs/batch_size)
train = pd_to_tfdata(full_data, p_ident, e_ident, b_ident, d_ident=d_ident, idx=idx_training)
train = train.cache().repeat().batch(batch_size).prefetch(batch_size)
callbacks = [
    keras.callbacks.EarlyStopping('loss', min_delta=small_batch_tol, patience=1, mode='min'),
    keras.callbacks.TensorBoard(log_dir='./output/temp/logs'),
    keras.callbacks.History()
]
print("Starting training with batch size {}".format(batch_size))
dn_model.fit(train, epochs=epoch, callbacks=callbacks, verbose=2, steps_per_epoch=steps_per_epoch)

# Large batch training for reduced variance around target area
batch_size = int(n_training_obs/10)
steps_per_epoch = int(idx_training.shape[0]/batch_size)
train = pd_to_tfdata(full_data, p_ident, e_ident, b_ident, d_ident=d_ident, idx=idx_training)
train = train.cache().repeat().batch(batch_size).prefetch(batch_size)
callbacks = [
    keras.callbacks.EarlyStopping('loss', min_delta=large_batch_tol, patience=1, mode='min'),
    keras.callbacks.TensorBoard(log_dir='./output/temp/logs'),
    keras.callbacks.History()
]
print("Starting training with batch size {}".format(batch_size))
dn_model.fit(train, epochs=epoch, callbacks=callbacks, verbose=2, steps_per_epoch=steps_per_epoch)

# Full training for minimum variance around target area
batch_size = n_training_obs
steps_per_epoch = int(idx_training.shape[0]/batch_size)
train = pd_to_tfdata(full_data, p_ident, e_ident, b_ident, d_ident=d_ident, idx=idx_training)
train = train.cache().repeat().batch(batch_size).prefetch(batch_size)
callbacks = [
    keras.callbacks.EarlyStopping('loss', min_delta=large_batch_tol, patience=1, mode='min'),
    keras.callbacks.TensorBoard(log_dir='./output/temp/logs'),
    keras.callbacks.History()
]
print("Starting training with batch size {}".format(batch_size))
dn_model.fit(train, epochs=epoch, callbacks=callbacks, verbose=2, steps_per_epoch=steps_per_epoch)

dn_model.