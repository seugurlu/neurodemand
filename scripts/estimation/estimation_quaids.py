"""
    Returns a dictionary of predicted coefficients for quaids models. Keys are sample keys.
"""
# Import Modules
import sys
sys.path.append(".\\scripts\\estimation\\")  # Add estimation scripts to the path for easy importing.

from joblib import Parallel, delayed
import demandtools as dt
import pandas as pd
import numpy as np
import function_quaids as fq

# Set Hyper-parameters and Other useful things.

with_demographic = False
number_cores = 6
n_bootstrap = 200
price_column_name_identifier = 'pt_'
expenditure_column_name_identifier = 'total_expenditure'
demographics_column_name_identifier = 'd_'
budget_share_column_name_identifier = 'wt_'
data_path = "./output/sample_adjustments/edited_data.csv"
data_index_column_name_identifier = 'index'
idx_bootstrap_path = "./output/sample_adjustments/idx_bootstrap.npy"


# Import Data

full_data = pd.read_csv(data_path, index_col=data_index_column_name_identifier)
idx_bootstrap = np.load(idx_bootstrap_path, allow_pickle=True).item()

# Proceed with estimation


def quaids_estimations(sample_key):
    idx_training = idx_bootstrap[sample_key]['training_sample']
    coefficients = fq.quaids_estimator(idx_training, full_data, with_demographic,
                                       price_column_name_identifier, expenditure_column_name_identifier,
                                       budget_share_column_name_identifier, demographics_column_name_identifier)
    return coefficients


output = Parallel(n_jobs=number_cores, verbose=10)(delayed(quaids_estimations)(sample_key)
                                                   for sample_key in range(n_bootstrap))
quaids_models = dict(list(zip(np.arange(n_bootstrap), output)))
np.save("./output/estimation/quaids_coefficients.npy", quaids_models)
