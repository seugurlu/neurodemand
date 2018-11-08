import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import demandtools as dt
import torch.nn as nn

# Initialize
with_demographic = False
price_column_name_identifier = 'pt_'
expenditure_column_name_identifier = 'total_expenditure'
demographics_column_name_identifier = 'd_'
budget_share_column_name_identifier = 'wt_'
data_path = "./output/sample_adjustments/edited_data.csv"
data_index_column_name_identifier = 'index'
aids_model_path = "./output/estimation/aids_coefficients.npy"
quaids_model_path = "./output/estimation/quaids_coefficients.npy"
dnn_model_path = "./output/estimation/dnn_estimates.npy"
idx_bootstrap_path = "./output/sample_adjustments/idx_bootstrap.npy"
n_bootstrap = 200
estimators = ['AIDS', 'QUAIDS', 'DNN']
mse_all = pd.DataFrame(index=np.arange(n_bootstrap), columns=estimators)
n_good = 10
if with_demographic:
    n_demographic = 3
else:
    n_demographic = 0


# Neural Network Specific Hyper-parameters
n_input_layer = n_good + n_demographic + 1
n_output_layer = n_good
hl_transformation = nn.ReLU()

# Import Estimated Models
aids_models = np.load(aids_model_path).item()
quaids_models = np.load(quaids_model_path).item()
nn_models = np.load(dnn_model_path).item()

# Import Data
full_data = pd.read_csv(data_path, index_col=data_index_column_name_identifier)
idx_bootstrap = np.load(idx_bootstrap_path).item()


# Obtain costs
def aids_obtain_cost(sample_key):
    # Import test samples
    test_set_index = idx_bootstrap[sample_key]['test_sample']
    test_sample = full_data.loc[test_set_index, :]
    price = test_sample.loc[:, test_sample.columns.str.startswith(price_column_name_identifier)]
    expenditure = test_sample.loc[:, test_sample.columns.str.startswith(expenditure_column_name_identifier)]
    budget_share = test_sample.loc[:, test_sample.columns.str.startswith(budget_share_column_name_identifier)]
    if with_demographic is True:
        demographics = test_sample.loc[:, test_sample.columns.str.startswith(demographics_column_name_identifier)]
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
    price = test_sample.loc[:, test_sample.columns.str.startswith(price_column_name_identifier)]
    expenditure = test_sample.loc[:, test_sample.columns.str.startswith(expenditure_column_name_identifier)]
    budget_share = test_sample.loc[:, test_sample.columns.str.startswith(budget_share_column_name_identifier)]
    if with_demographic is True:
        demographics = test_sample.loc[:, test_sample.columns.str.startswith(demographics_column_name_identifier)]
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


def dnn_obtain_cost(sample_key):
    test_set_index = idx_bootstrap[sample_key]['test_sample']
    test_sample = full_data.loc[test_set_index, :]
    price = test_sample.loc[:, test_sample.columns.str.startswith(price_column_name_identifier)]
    expenditure = test_sample.loc[:, test_sample.columns.str.startswith(expenditure_column_name_identifier)]
    budget_share = test_sample.loc[:, test_sample.columns.str.startswith(budget_share_column_name_identifier)]
    if with_demographic is True:
        demographics = test_sample.loc[:, test_sample.columns.str.startswith(demographics_column_name_identifier)]
        test_data_input = dt.DemandDataset(price, expenditure, budget_share, demographics)
    else:
        test_data_input = dt.DemandDataset(price, expenditure, budget_share)
    y_test = test_data_input.y
    x_test = test_data_input.x
    n_hidden_layer = nn_models[sample_key][0]  # Obtain Hyper-parameter
    dnn_model = dt.network_model(n_input_layer, n_hidden_layer, n_output_layer, hl_transformation)  # Define model
    parameters = nn_models[sample_key][1]['trained_model'][n_hidden_layer - 2].state_dict()  # Obtain trained model
    # Fix keys in parameters dictionary
    parameters['hl1.weight'] = parameters.pop("0.hl1.weight")
    parameters['hl1.bias'] = parameters.pop('0.hl1.bias')
    parameters['ol.weight'] = parameters.pop("0.ol.weight")
    parameters['ol.bias'] = parameters.pop('0.ol.bias')
    dnn_model.load_state_dict(parameters)
    loss = nn.MSELoss()
    y_prediction = dnn_model(x_test)
    test_mse = loss(y_prediction, y_test)
    return test_mse.item()


for sample_key in range(n_bootstrap):
    mse_all.loc[sample_key, 'AIDS'] = aids_obtain_cost(sample_key)
    mse_all.loc[sample_key, 'QUAIDS'] = quaids_obtain_cost(sample_key)
    mse_all.loc[sample_key, 'DNN'] = dnn_obtain_cost(sample_key)

mse_all.plot.kde()
plt.show(block=True)
