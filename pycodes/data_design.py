import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.chdir("/home/seugurlu/Desktop/SVN/trunk/estimering") #Personal Comp - Virtual Comp Directory

#Read the dataset and set hyperparameters
main_sample = pd.read_csv('data/raw/estimation_data.txt')
main_sample.rename(columns = {'Unnamed: 0' : 'hhid'}, inplace = True)
main_sample.to_hdf('./data/processed/main_sample', 'main_sample')
number_hh = main_sample.shape[0]
number_goods = len(main_sample.filter(regex = "^p").columns)
number_bootstrap = 1000

"""Generate Training/CV/Test Samples for bootstrap"""
for i in range(number_bootstrap):
    bootstrap_sample = main_sample.sample(frac = 1, replace = True, random_state = i)
    x_ = bootstrap_sample.iloc[:,1:number_goods+2].astype(np.float32)
    y_ = bootstrap_sample.iloc[:,number_goods+2:2*number_goods+2].astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.30, random_state=(i+1)*number_bootstrap)
    x_cv, x_test, y_cv, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = (i+1)*number_bootstrap + 1)
    save_name = 'bootstrap_sample'+str(i+1)
    bootstrap_sample.to_hdf('data/processed/'+save_name,'bootstrap_sample')
    x_.to_hdf('data/processed/'+save_name,'x_')
    y_.to_hdf('data/processed/'+save_name,'y_')
    x_train.to_hdf('data/processed/'+save_name,'x_train')
    y_train.to_hdf('data/processed/'+save_name,'y_train')
    x_cv.to_hdf('data/processed/'+save_name,'x_cv')
    y_cv.to_hdf('data/processed/'+save_name,'y_cv')
    x_test.to_hdf('data/processed/'+save_name,'x_test')
    y_test.to_hdf('data/processed/'+save_name,'y_test')