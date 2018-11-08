"""
    Returns a dictionary of predicted coefficients for quaids models. Keys are sample keys.
"""
# Import Modules
from joblib import Parallel, delayed
import demandtools as dt
import pandas as pd
import numpy as np


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
idx_bootstrap = np.load(idx_bootstrap_path).item()


# Proceed with estimation


def quaids_estimations(sample_key):
    train_set_index = idx_bootstrap[sample_key]['training_sample']
    train = full_data.loc[train_set_index, :]
    price = train.loc[:, train.columns.str.startswith(price_column_name_identifier)]
    expenditure = train.loc[:, train.columns.str.startswith(expenditure_column_name_identifier)]
    budget_share = train.loc[:, train.columns.str.startswith(budget_share_column_name_identifier)]
    if with_demographic is True:
        demographics = train.loc[:, train.columns.str.startswith(demographics_column_name_identifier)]
        quaids = dt.Quaids(price, expenditure, budget_share, demographics)
    else:
        quaids = dt.Quaids(price, expenditure, budget_share)
    quaids.optimize()
    if with_demographic is True:
        coefficients = dict([('alpha', quaids.alpha), ('beta', quaids.beta),
                            ('gamma', quaids.gamma), ('alpha_demographic', quaids.alpha_demographic),
                            ('lambda', quaids.lambdas)])
    else:
        coefficients = dict([('alpha', quaids.alpha), ('beta', quaids.beta),
                            ('gamma', quaids.gamma), ('lambda', quaids.lambdas)])  # Uncomment for demographics
    return coefficients


output = Parallel(n_jobs=number_cores, verbose=10)(delayed(quaids_estimations)(sample_key)
                                                   for sample_key in range(n_bootstrap))
quaids_models = dict(list(zip(np.arange(n_bootstrap), output)))
np.save("./output/estimation/quaids_coefficients.npy", quaids_models)
