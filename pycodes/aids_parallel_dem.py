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

import aids_estimation_viaR_dem as aids

#Hyperparameters
number_bootstrap = 1000
number_cores = 2 #Main comp virtual
#number_cores = 35 #Server

#Program Parallel Loop
def parallel_aids (sample_key):    
    data = pd.HDFStore(r'data/processed/bootstrap_sample'+str(sample_key+1), 'r')
    x_train = data['x_train_dem']
    y_train = data['y_train']
    x_train = x_train.iloc[:,:-1]
    
    result = aids.aids(x_train, y_train)
    np.save(os.getcwd()+'/output/AIDS/Demographics/aids_dem_'+str(sample_key+1), result)
    data.close()
    
Parallel(n_jobs = number_cores)(delayed(parallel_aids)(s) for s in range(number_bootstrap) )
