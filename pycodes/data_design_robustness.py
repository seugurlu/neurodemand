#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:47:14 2018

@author: seugurlu

@info: this file creates bootstrapped datasets for robustness check to sample creation. In the main text, the data is created by
first bootstrapping and then dividing each sample to estimation/cv/test samples. This may create the following problem: in cv and
test samples, there will be observations that were used in the estimation sample.

As a robustness test to this issue, this code generates samples with first splitting and then bootstrapping an estimation sample.
CV and test samples are not bootstrapped.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.chdir("/home/seugurlu/Desktop/SVN/trunk/estimering") #Personal Comp - Virtual Comp Directory

#Read the dataset and set hyperparameters
main_sample = pd.read_csv('data/raw/estimation_data.txt')
main_sample.rename(columns = {'Unnamed: 0' : 'hhid'}, inplace = True)
main_sample.to_hdf('./data/processed/robustness_test_samples', 'main_sample') #Open an hdf file with main sample
number_hh = main_sample.shape[0]
number_goods = len(main_sample.filter(regex = "^p").columns)
number_bootstrap = 200
rng = 0

"""Generate Training/CV/Test Samples for bootstrap"""
x_ = main_sample.iloc[:,1:number_goods+2].astype(np.float32)
y_ = main_sample.iloc[:,number_goods+2:2*number_goods+2].astype(np.float32)
rng = rng + 1
x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.30, random_state = rng)
rng = rng+1
x_cv, x_test, y_cv, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = rng)

x_.to_hdf('data/processed/sample_robustness_data/main_data', 'x_') #Save main x sample
y_.to_hdf('data/processed/sample_robustness_data/main_data', 'y_') #Save main y sample

for i in np.arange(number_bootstrap)+1: #Generate bootstrapped training sets and save to the hdf file
    rng = rng + 1
    bootstrapped_x_train = x_train.sample(frac = 1, replace = True, random_state = rng)
    bootstrapped_y_train = y_train.loc[bootstrapped_x_train.index.values,:]
    bootstrapped_x_train.to_hdf('data/processed/sample_robustness_data/sample'+str(i), 'x_train' )
    bootstrapped_y_train.to_hdf('data/processed/sample_robustness_data/sample'+str(i), 'y_train' )
    
    rng = rng + 1
    bootstrapped_x_cv = x_cv.sample(frac = 1, replace = True, random_state = rng)
    bootstrapped_y_cv = y_cv.loc[bootstrapped_x_cv.index.values,:]
    bootstrapped_x_cv.to_hdf('data/processed/sample_robustness_data/sample'+str(i), 'x_cv' )
    bootstrapped_y_cv.to_hdf('data/processed/sample_robustness_data/sample'+str(i), 'y_cv' )
    
    rng = rng + 1
    bootstrapped_x_test = x_test.sample(frac = 1, replace = True, random_state = rng)
    bootstrapped_y_test = y_test.loc[bootstrapped_x_test.index.values,:]
    bootstrapped_x_test.to_hdf('data/processed/sample_robustness_data/sample'+str(i), 'x_test' )
    bootstrapped_y_test.to_hdf('data/processed/sample_robustness_data/sample'+str(i), 'y_test' )