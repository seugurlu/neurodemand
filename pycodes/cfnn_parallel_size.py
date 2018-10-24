# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:18:45 2017

@author: s12334
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:52:40 2017
This is the main code for running and storing cfnn codes in parallel.
@author: seugurlu
"""
from sys import path
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import os
import neural_functions as nf

os.chdir("/home/serhat/neurodemand/estimering") #Server
#os.chdir("/home/seugurlu/Desktop/SVN/trunk/estimering") #School Comp - Virtual Comp
path.append(os.getcwd()+'/scripts/pycodes')

import cfnn_functions_size as cffn

#Hyperparameters
number_bootstrap = 100
#number_cores = 2
number_cores = 35
tol_coef = 1e-4
tol_iter = 1e-6
number_goods = 10

#Program Parallel Loop
def parallel_nfnn (sample_key):    
    data = pd.HDFStore(r'data/processed/bootstrap_sample'+str(sample_key+1), 'r')
    x_train = data['x_train']
    y_train = data['y_train']
    x_cv = data['x_cv']
    y_cv = data['y_cv']
    x_test = data['x_test']
    y_test = data['y_test']
    
    x_cv = np.log( nf.impose_homogeneity( x_cv ) )#Generate input matrix as homogeneity adjusted log
    x_test = np.log( nf.impose_homogeneity( x_test ) )#Generate input matrix as homogeneity adjusted log
    
    for sample_size in [1000, 2000, 3000, 4000]:
        x_train_sample = x_train[0:sample_size]
        y_train_sample = y_train[0:sample_size]
    
        #Adjust data for homogeneity and log
        x_train_sample = np.log( nf.impose_homogeneity ( x_train_sample ) )#Generate input matrix as homogeneity adjusted log
    
        cv_accuracy, test_accuracy, w_1_out, w_2_out, b_1_out, b_2_out, hh_size = cffn.main_with_cv(sample_key+1, x_train_sample, y_train_sample, x_cv, y_cv, x_test, y_test, tol_iter, tol_coef, number_goods)
        np.savez(os.getcwd()+"/output/NN/Size/pycoefficients_cfnn_"+str(sample_key+1)+'_'+str(sample_size), w_1 = w_1_out, w_2 = w_2_out, b_1 = b_1_out, b_2 = b_2_out, 
             hh_size = hh_size, cv_accuracy = cv_accuracy, test_accuracy = test_accuracy)
        
    data.close()
    
Parallel(n_jobs = number_cores)(delayed(parallel_nfnn)(s) for s in range(number_bootstrap))

#with np.load('./output/NN/pycoefficients3.npz') as data:
#            w_1_out = data['w_1']
#            w_2_out = data['w_2']
#            b_1_out = data['b_1']
#            b_2_out = data['b_2']
#            hh_size = data['hh_size']
#            cv_accuracy = data['cv_accuracy']
#            test_accuracy = data['test_accuracy']
