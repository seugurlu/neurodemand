from tensorflow import keras as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import demandtools as dt
import seaborn as sns
import sys
sys.path.append('./scripts/estimation/')
import network_functions as nf

# Initialize
with_demographic = False
p_ident = 'pt_'
e_ident = 'total_expenditure'
d_ident = None
b_ident = 'wt_'
data_path = "./output/sample_adjustments/edited_data.csv"
data_index_column_name_identifier = 'index'
aids_model_path = "./output/estimation/aids_coefficients.npy"
quaids_model_path = "./output/estimation/quaids_coefficients.npy"
dn_model_path = "./output/estimation/dn/"
idx_bootstrap_path = "./output/sample_adjustments/idx_bootstrap.npy"
n_bootstrap = 200
estimators = ['AIDS', 'QUAIDS', 'DN']
mse_all = pd.DataFrame(index=np.arange(n_bootstrap), columns=estimators)
n_good = 10
if with_demographic:
    n_demographic = 3
else:
    n_demographic = 0

# Import Estimated AIDS and QUAIDS Models
aids_models = np.load(aids_model_path).item()
quaids_models = np.load(quaids_model_path).item()

# Import Data
full_data = pd.read_csv(data_path, index_col=data_index_column_name_identifier)
idx_bootstrap = np.load(idx_bootstrap_path).item()


# Obtain costs
def aids_obtain_cost(sample_key):
    # Import test samples
    test_set_index = idx_bootstrap[sample_key]['test_sample']
    test_sample = full_data.loc[test_set_index, :]
    price = test_sample.loc[:, test_sample.columns.str.startswith(p_ident)]
    expenditure = test_sample.loc[:, test_sample.columns.str.startswith(e_ident)]
    budget_share = test_sample.loc[:, test_sample.columns.str.startswith(b_ident)]
    if with_demographic is True:
        demographics = test_sample.loc[:, test_sample.columns.str.startswith(d_ident)]
        aids = dt.Aids(price, expenditure, budget_share, demographics)
    else:
        aids = dt.Aids(price, expenditure, budget_share)

    # Import coefficients
    alpha = aids_models[sample_key]['alpha']
    beta = aids_models[sample_key]['beta']
    gamma = aids_models[sample_key]['gamma']
    if with_demographic is True:
        alpha_demographic = aids_models[sample_key]['alpha_demographic']
        aids.external_coefficient(alpha, beta, gamma, alpha_demographic)
    else:
        aids.external_coefficient(alpha, beta, gamma)
    return ((aids.budget_share - aids.predict())**2).mean()


def quaids_obtain_cost(sample_key):
    # Import test samples
    test_set_index = idx_bootstrap[sample_key]['test_sample']
    test_sample = full_data.loc[test_set_index, :]
    price = test_sample.loc[:, test_sample.columns.str.startswith(p_ident)]
    expenditure = test_sample.loc[:, test_sample.columns.str.startswith(e_ident)]
    budget_share = test_sample.loc[:, test_sample.columns.str.startswith(b_ident)]
    if with_demographic is True:
        demographics = test_sample.loc[:, test_sample.columns.str.startswith(d_ident)]
        quaids = dt.Quaids(price, expenditure, budget_share, demographics)
    else:
        quaids = dt.Quaids(price, expenditure, budget_share)

    # Import coefficients
    alpha = quaids_models[sample_key]['alpha']
    beta = quaids_models[sample_key]['beta']
    gamma = quaids_models[sample_key]['gamma']
    lambdas = quaids_models[sample_key]['lambda']
    if with_demographic is True:
        alpha_demographic = quaids_models[sample_key]['alpha_demographic']
        quaids.external_coefficient(alpha, beta, gamma, lambdas, alpha_demographic)
    else:
        quaids.external_coefficient(alpha, beta, gamma, lambdas)
    return ((quaids.budget_share - quaids.predict())**2).mean()

def dn_obtain_cost(sample_key):
    idx_test = idx_bootstrap[sample_key]['test_sample']  # Test sample.
    x_test, y_test = nf.prepare_data(full_data, p_ident, e_ident, b_ident, d_ident=d_ident, idx=idx_test)  # x and y.
    model_path = dn_model_path + "sample_" + str(sample_key) + ".h5"
    K.backend.clear_session()  # Clear session before loading the model.
    model = K.models.load_model(model_path)  # Load model.
    test_cost = model.evaluate(x=x_test, y=y_test, verbose=0)
    return test_cost

for sample_key in range(n_bootstrap):
    mse_all.loc[sample_key, 'AIDS'] = aids_obtain_cost(sample_key)
    mse_all.loc[sample_key, 'QUAIDS'] = quaids_obtain_cost(sample_key)
    mse_all.loc[sample_key, 'DN'] = dn_obtain_cost(sample_key)

mse_all = mse_all.astype(float)  # Turn columns to floats for boxplot
boxplot = mse_all.boxplot()
