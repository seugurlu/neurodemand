#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 18:52:50 2017

@author: seugurlu
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
#os.chdir("/home/serhat/neurodemand/estimering") #Server
path.append(os.getcwd()+'/scripts/pycodes')

import quaids_estimation_viaR as quaids

#Hyperparameters
number_bootstrap = 100
number_cores = 3 #Main comp virtual
#number_cores = 35 #Server

#Program Parallel Loop
def parallel_quaids (sample_key):    
    data = pd.HDFStore(r'data/processed/bootstrap_sample'+str(sample_key+1), 'r')
    x_train = data['x_train']
    y_train = data['y_train']
    
    for sample_size in [1000, 2000, 3000, 4000]:
        x_train_sample = x_train[0:sample_size]
        y_train_sample = y_train[0:sample_size]
        result = quaids.quaids(x_train_sample, y_train_sample)
        np.save(os.getcwd()+'/output/QUAIDS/Sample Sizes/quaids'+str(sample_key+1)+'_'+str(sample_size), result)
        
    data.close()
    
Parallel(n_jobs = number_cores)(delayed(parallel_quaids)(s) for s in range(number_bootstrap) )