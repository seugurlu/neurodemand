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


os.chdir("/home/seugurlu/Desktop/SVN/trunk/estimering") #Personal Comp - Virtual Comp Directory
#os.chdir("/home/serhat/SVN/trunk/estimering") #Server
path.append(os.getcwd()+'/scripts/pycodes')

import nfnn_dem_functions as nffn

#Hyperparameters
number_bootstrap = 1000
number_cores = 35
tol_coef = 1e-4
tol_iter = 1e-6
number_goods = 10

#Program Parallel Loop
def parallel_nfnn (sample_key):    
    data = pd.HDFStore(r'data/processed/bootstrap_sample'+str(sample_key+1), 'r')
    x_train = data['x_train_dem']
    y_train = data['y_train']
    x_cv = data['x_cv_dem']
    y_cv = data['y_cv']
    x_test = data['x_test_dem']
    y_test = data['y_test']
    
    #Adjust data for homogeneity and log
    x_train.iloc[:,:number_goods+1] = np.log( x_train.iloc[:,:number_goods+1] )#Generate input matrix as log
    x_cv.iloc[:,:number_goods+1] = np.log( x_cv.iloc[:,:number_goods+1] )#Generate input matrix as log
    x_test.iloc[:,:number_goods+1] = np.log( x_test.iloc[:,:number_goods+1] )#Generate input matrix as log
    
    #Drop d.hh.size = d.n.child + d.n.adult
    x_train = x_train.iloc[:,:-1]
    x_cv = x_cv.iloc[:,:-1]
    x_test = x_test.iloc[:,:-1]
    
    nffn.main(sample_key+1, x_train, y_train, x_cv, y_cv, x_test, y_test, tol_iter, number_goods)
    data.close()
    
Parallel(n_jobs = number_cores)(delayed(parallel_nfnn)(s) for s in range(500) ) #500, 500:750, 750:1000
