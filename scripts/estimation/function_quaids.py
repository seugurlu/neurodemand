"""AIDS functions"""
import demandtools as dt
import pandas as pd
import numpy as np


def quaids_estimator(idx, full_data, with_demographic=True,
					 price_column_name_identifier='p', expenditure_column_name_identifier='total_expenditure',
					 budget_share_column_name_identifier='w', demographics_column_name_identifier='d'):

	"""Extract training data and perform estimation."""

	"""Input check"""
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
		quaids = dt.Quaids(price, expenditure, budget_share, demographics)  # Define QUAIDS model
	else:
		quaids = dt.Quaids(price, expenditure, budget_share)

	"""Optimization"""
	quaids.optimize()

	"""Extract estimated coefficients"""
	if with_demographic is True:
		coefficients = dict([('alpha', quaids.alpha), ('beta', quaids.beta),
							 ('gamma', quaids.gamma), ('alpha_demographic', quaids.alpha_demographic),
							 ('lambda', quaids.lambdas)])
	else:
		coefficients = dict([('alpha', quaids.alpha), ('beta', quaids.beta),
							 ('gamma', quaids.gamma), ('lambda', quaids.lambdas)])  # Uncomment for demographics
	return coefficients
