"""
    Divides the main sample into train, cross-validation, test samples. Applies transformations.
    Returns a dictionary of idx for bootstrap samples.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Hyperparameters
file_path = './raw_data/estimation_data.csv'
index_col_name = 'index'
price_column_name_identifier = 'pt_'
expenditure_column_name_identifier = 'total_expenditure'
demographics_column_name_identifier = 'd_'
budget_share_column_name_identifier = 'wt_'
seed = 1809211825  # YYMMDDHHMM
n_bootstrapped_samples = 200
training_size_fraction = 0.7

#  Read data
main_data = pd.read_csv(file_path, index_col=index_col_name)

#  Apply adjustments
price = main_data.loc[:, main_data.columns.str.startswith(price_column_name_identifier)].apply(np.log)
expenditure = main_data.loc[:, expenditure_column_name_identifier].apply(np.log)
budget_share = main_data.loc[:, main_data.columns.str.startswith(budget_share_column_name_identifier)]
demographics = main_data.loc[:, main_data.columns.str.startswith(demographics_column_name_identifier)]
demographics_mean = demographics.mean()  # Record mean for each column
demographics_std = demographics.std()  # Record std for each column
demographics = (demographics-demographics_mean)/demographics_std
edited_data = pd.concat([price, expenditure, demographics, budget_share], axis=1)
edited_data = shuffle(edited_data, random_state=seed)
len_train = int(edited_data.shape[0] * training_size_fraction)


# Draw samples
np.random.seed(seed+1)
bootstrap_sample_train = np.random.choice(np.array(edited_data.index), size=len_train*n_bootstrapped_samples)
idx_bootstrap = {}
for i in range(n_bootstrapped_samples):
    idx_bootstrap[i] = {}
    idx_bootstrap[i]['training_sample'] = bootstrap_sample_train[i*len_train:(i+1)*len_train]
    idx_bootstrap_sample_other = np.in1d(np.array(edited_data.index), idx_bootstrap[i]['training_sample'])
    bootstrap_sample_other = np.array(edited_data.loc[~idx_bootstrap_sample_other, :].index)
    # Dividing this sample to two is random because of the initial random shuffle.
    idx_bootstrap[i]['cv_sample'] = bootstrap_sample_other[:int(bootstrap_sample_other.shape[0]/2)]
    idx_bootstrap[i]['test_sample'] = bootstrap_sample_other[int(bootstrap_sample_other.shape[0]/2):]

# Prepare Output
normalization_statistics = dict([('means', demographics_mean),
                                 ('stds', demographics_std)])

# Export results
np.save('./output/sample_adjustments/norm_stats.npy', normalization_statistics)
np.save('./output/sample_adjustments/idx_bootstrap.npy', idx_bootstrap)
edited_data.to_csv('./output/sample_adjustments/edited_data.csv')

