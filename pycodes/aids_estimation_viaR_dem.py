#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 19:07:05 2017
This code connects R, estimates obtains AIDS and QUAIDS estimates, and saves.
@author: seugurlu
"""
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import numpy as np

def aids(x, y):
    pandas2ri.activate()
    demand_tools = importr('demand.tools')
    demand_system = demand_tools.demand_system
    
    log_price = np.log( x.filter(regex = r'^p.', axis = 1) )
    log_expenditure = np.log( x['total.expenditure'] )
    demographics = x.filter(regex = r'^d.', axis = 1)
    
    log_price_r = pandas2ri.py2ri(log_price)
    log_expenditure_r = pandas2ri.py2ri(log_expenditure)
    demographics_r = pandas2ri.py2ri(demographics)
    y_r = pandas2ri.py2ri(y)
    result = demand_system(log_price_r, log_expenditure_r, y_r, dem = demographics_r, model = "AIDS", maxIter = 1000)
    
    return(result)