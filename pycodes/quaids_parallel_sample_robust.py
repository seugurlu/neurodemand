#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:22:32 2018

@author: seugurlu
"""

from sys import path
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import os

try:
    os.chdir("/home/seugurlu/Desktop/SVN/trunk/estimering") #Personal Comp - Virtual Comp Directory
except:
    os.chdir("/home/serhat/SVN/trunk/estimering") #Server

path.append(os.getcwd()+'/scripts/pycodes')

import quaids_estimation_viaR as quaids

#Hyperparameters
number_bootstrap = 200
number_cores = 4 #Main comp virtual
#number_cores = 35 #Server

def parallel_quaids (sample_key):        
    data = pd.HDFStore(r'data/processed/sample_robustness_data/sample'+str(sample_key), 'r')
    x_train = data['x_train']
    y_train = data['y_train']
    result = quaids.quaids(x_train, y_train)
    np.save(os.getcwd()+'/output/robustness/sample splitting/quaids/quaids'+str(sample_key), result)
    
Parallel(n_jobs = number_cores)(delayed(parallel_quaids)(s) for s in np.arange(number_bootstrap)+1 )
