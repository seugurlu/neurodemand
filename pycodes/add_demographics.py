#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:45:01 2017

@author: seugurlu
"""
import pandas as pd
from sys import path
import os

#os.chdir("/home/seugurlu/Desktop/SVN/trunk/estimering") #Personal Comp - Virtual Comp Directory
#os.chdir("/home/serhat/neurodemand/estimering") #Server
os.chdir("/home/seugurlu/Desktop/SVN/neurodemand/trunk/estimering") #School Comp - Virtual Comp
path.append(os.getcwd()+'/scripts/pycodes')

number_bootstrap = 1000
main_data = pd.read_csv("./data/raw/estimation_data.txt")
index_size = len(main_data)

for bootstrap_sample in range(number_bootstrap):
    data = pd.HDFStore(r'data/processed/bootstrap_sample'+str(bootstrap_sample+1), 'r+')
    x_train = data['x_train']
    x_cv = data['x_cv']
    x_test = data['x_test']
    save_name = 'bootstrap_sample'+str(bootstrap_sample+1)
    
    x_train_dem = pd.concat([x_train, main_data.filter(regex = "^d")], axis = 1, join = 'inner')
    x_cv_dem = pd.concat([x_cv, main_data.filter(regex = "^d")], axis = 1, join = 'inner')
    x_test_dem = pd.concat([x_test, main_data.filter(regex = "^d")], axis = 1, join = 'inner')
    
    x_train_dem.to_hdf('data/processed/'+save_name,'x_train_dem')
    x_cv_dem.to_hdf('data/processed/'+save_name,'x_cv_dem')
    x_test_dem.to_hdf('data/processed/'+save_name,'x_test_dem')
    
    data.close()