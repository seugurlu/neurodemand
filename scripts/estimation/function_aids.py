"""AIDS functions"""
import demandtools as dt
import pandas as pd
import numpy as np

def aids_estimator(sample_key, idx, full_data, with_demographic=True,
					 price_column_name_identifier='p', expenditure_column_name_identifier='total_expenditure',
					 budget_share_column_name_identifier='w', demographics_column_name_identifier='d'):

	"""Extract training data and perform estimation."""

	"""Input check"""
	assert isinstance(sample_key, (int, np.integer)), "Sample key must be an integer."
	assert isinstance(idx, (list, np.ndarray, pd.Series)), "idx must be a list, numpy array, or pandas series."
	assert(all(map(lambda x: isinstance(x, (int, np.integer)), idx))), "idx elements must be a list of integers."
	assert isinstance(full_data, pd.DataFrame), "full_data must be a pandas dataframe."
	assert isinstance(with_demographic, bool), "with_demographic must be True or False."
	assert all(map(lambda x: isinstance(x, str),
				   [price_column_name_identifier, expenditure_column_name_identifier,
					budget_share_column_name_identifier,
					demographics_column_name_identifier])), "Column identifiers must be string."

	"""Extract training prices, expenditures, and demographics."""
	training_sample = full_data.loc[idx, :]
	price = training_sample.loc[:, training_sample.columns.str.startswith(price_column_name_identifier)]
	expenditure = training_sample.loc[:, training_sample.columns.str.startswith(expenditure_column_name_identifier)]
	budget_share = training_sample.loc[:, training_sample.columns.str.startswith(budget_share_column_name_identifier)]
	if with_demographic is True:
		demographics = training_sample.loc[:, training_sample.columns.str.startswith(demographics_column_name_identifier)]
		aids = dt.Aids(price, expenditure, budget_share, demographics)  # Define AIDS model
	else:
		aids = dt.Aids(price, expenditure, budget_share)  # Define AIDS model

	"""Optimization"""
	aids.optimize()

	"""Extract estimated coefficients"""
	if with_demographic is True:
		coefficients = dict([('alpha', aids.alpha), ('beta', aids.beta),
							 ('gamma', aids.gamma), ('alpha_demographic', aids.alpha_demographic)])
	else:
		coefficients = dict([('alpha', aids.alpha), ('beta', aids.beta), ('gamma', aids.gamma)])
	return coefficients
