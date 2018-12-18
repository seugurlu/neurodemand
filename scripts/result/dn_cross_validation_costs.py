'''Code to analyze cross-validation results for DN estimates and select preferred
model specifications.'''
# Import packages.
from tensorflow import keras as K
import sys
sys.path.append('./scripts/estimation/')
import network_functions as nf
import numpy as np
import pandas as pd

# Hyperparameters.
p_ident = 'pt_'  # String identifier for price columns.
e_ident = 'total_expenditure'  # String identifier for total expenditure column.
b_ident = 'wt_'  # String identifier for budget share columns.
d_ident = None  # String identifier for demographic variables columns. Set to 'd_' for estimation with demographics.
file_path = "./output/sample_adjustments/edited_data.csv"  # Path to the training data.
data_index_column_name_identifier = 'index'  # Column name that holds data point indices in input files.
idx_bootstrap_data_path = "./output/sample_adjustments/idx_bootstrap.npy"  # Path to bootstrap indices.
n_node_start = 5 # Take from estimation file or saved files.
n_node_end = 15 # Take from estimation file or saved files.
n_bootstrap_samples = 200 # Number of bootstrapped samples.
save_path = './output/analysis/'

# Load data.
main_data = pd.read_csv(file_path, index_col=data_index_column_name_identifier) # Load main data set.
idx_bootstrap = np.load(idx_bootstrap_data_path).item() # Observation indices for each bootstrapped sample.

# Generate table to store cv-losses.
sample_key_range = list(range(n_bootstrap_samples)) # List of an integer sequence to ID sample key.
n_node_range = list(range(n_node_start, n_node_end+1)) # List of an integer sequence to ID n_node.
table_index = pd.MultiIndex.from_product([sample_key_range, n_node_range],
                                         names=['bootstrap_sample', 'n_node']) # Unique ID:sample key + n_node
cv_losses = pd.DataFrame(index=table_index, columns=['cv_loss']) # Initiate table to store cv losses.


# Import output for the first bootstrapped sample.
for sample_key in range(n_bootstrap_samples): # Loop over sample keys.
    idx_cv = idx_bootstrap[sample_key]['cv_sample'] # Cross-validation observations.
    x_cv, y_cv = nf.prepare_data(main_data, p_ident, e_ident, b_ident, d_ident=d_ident, idx=idx_cv) # Obtain x and y.

    for node in range(n_node_start, n_node_end+1): # Loop over n_nodes.
        model_path = './output/temp/dn/cross_validation/sample_0_node_' + str(node) + '.h5' # Path to the relevant model.
        K.backend.clear_session() # Clear session before loading the model.
        model = K.models.load_model(model_path) # Load model.
        cv_losses.loc[(sample_key, node), 'cv_loss'] = model.evaluate(x=x_cv, y=y_cv, verbose=0) # Evaluate on cv set.

# Output cv losses due to time it takes to store the losses.
cv_losses.to_csv(save_path+'cv_losses.csv')