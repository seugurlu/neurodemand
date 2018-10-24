#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 22:21:55 2017
This file is the output script for the paper.
@author: seugurlu
"""
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
numpy2ri.activate()
pandas2ri.activate()

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sys import path
try:
    os.chdir("/home/seugurlu/Desktop/SVN/trunk/estimering")
except: 
    os.chdir("/home/serhat/SVN/trunk/estimering") #Server
    
path.append(os.getcwd()+'/scripts/pycodes')
import tensorflow as tf
import neural_functions as nf
import demand_predict as dp
sess = tf.InteractiveSession()

number_goods = 10
number_bootstrap = 200
number_cores = 35

#Set R functions
pracma = importr('pracma')
ode23 = pracma.ode23
bulirsch_stoer = pracma.bulirsch_stoer

#Define functions, set arrays
def cost(y_test, y_hat):
    return( 
            np.square( np.subtract(y_test, y_hat) ).stack().mean() 
            )

cost_frame = pd.DataFrame(columns = ['AIDS', 'QUAIDS', 'NFNN', 'CFNN'])
sample_key = 1
#%%
main_data = pd.HDFStore(r'data/processed/main_sample', 'r')['main_sample']

log_price = np.log( main_data.filter(regex = r'^p.', axis = 1) )
log_expenditure = np.log( main_data['total.expenditure'] )
log_inputs = log_price.join(log_expenditure).astype(np.float32)
#log_inputs = log_inputs.sample(n = 200, random_state = 100)

obs = np.array(log_inputs[0:1]).ravel()

aids_coef = np.array(np.load(r"./output/AIDS/aids"+str(sample_key)+".npy", encoding = 'bytes')[0])
quaids_coef = np.array(np.load(r"./output/QUAIDS/quaids"+str(sample_key)+".npy", encoding = 'bytes')[0])
with np.load(r'./output/NN/cfnn/pycoefficients_cfnn'+str(sample_key)+'.npz', 'r') as cfnn:
        w_1 = cfnn['w_1']
        w_2 = cfnn['w_2']
        b_1 = cfnn['b_1']
        b_2 = cfnn['b_2']
    
change_percent = 1.10
price_0 = np.exp(obs[:-1])
price_change = np.repeat(1.,number_goods)
price_change[0] = change_percent
price_1 = price_change * price_0
t = 1; y = 0.

def price(t):
    return( (price_1-price_0)*t + price_0)
#%%
def f_aids(t,y):
    func_input = np.append( price(t), np.exp(obs[-1]) - y ) 
    log_p = np.log(func_input[:-1])
    log_exp = np.log(func_input[-1])
    pred = dp.demand_predict(aids_coef, log_p, log_exp, number_goods, model = 'AIDS')
    q = pred * np.exp(log_exp) / np.exp(log_p)
    if t==0 :
        delta_p = np.zeros([1,number_goods])
    else:
        delta_p = price_1 - price_0
    ev = - np.matmul(q, np.transpose(delta_p))
    return(ev)
    
def f_quaids(t,y):
    func_input = np.append( price(t), np.exp(obs[-1]) - y ) 
    log_p = np.log(func_input[:-1])
    log_exp = np.log(func_input[-1])
    pred = dp.demand_predict(quaids_coef, log_p, log_exp, number_goods, model = 'QUAIDS')
    q = pred * np.exp(log_exp) / np.exp(log_p)
    if t==0 :
        delta_p = np.zeros([1,number_goods])
    else:
        delta_p = price_1 - price_0
    ev = - np.matmul(q, np.transpose(delta_p))
    return(ev)

def f_cfnn(t,y):
    func_input = np.append( price(t), np.exp(obs[-1]) - y ) 
    pred_input = np.log(func_input)
    pred,_ = nf.neural_predict(pred_input, w_1, w_2, b_1, b_2)
    q = pred * func_input[-1] / func_input[:-1]
    if t==0 :
        delta_p = np.zeros([1,number_goods])
    else:
        delta_p = price_1 - price_0
    ev = - np.matmul(q, np.transpose(delta_p))
    return(ev)
    
y0 = 0.

#AIDS
t_aids = np.array(ode23(f_aids, 1., 0., y0)[0])
ev_aids = bulirsch_stoer(f_aids, t_aids, y0, tol = 1e-4)[-1]

#QUAIDS
t_quaids = np.array(ode23(f_quaids, 1, 0, y0)[0])
ev_quaids = bulirsch_stoer(f_quaids, t_quaids, y0, tol = 1e-4)[-1]

#CFNN
t_cfnn = np.array(ode23(f_cfnn, 1, 0, y0)[0])
ev_cfnn = bulirsch_stoer(f_cfnn, t_cfnn, y0, tol = 1e-4)[-1]

#%%calculate from utilities
def ev_util(coef, price_1, price_0, number_goods, model = 'AIDS'):
    if model == 'AIDS':
        lna_new, b_new = dp.price_index(coef, np.log(price_1), number_goods, model = "AIDS")
        utility_new = (  obs[-1] - lna_new ) / b_new
        lna_old, b_old = dp.price_index(aids_coef, np.log(price_0), number_goods, model = "AIDS")
        lexp = utility_new * b_old + lna_old
        ev_util = np.exp(obs[-1]) - np.exp(lexp)
        return(ev_util)
    if model == 'QUAIDS':
        _,_,_,lambd = dp.unroll_coef(coef, number_goods, model = 'QUAIDS')
        lambda_new = (lambd*np.log(price_1)).sum()
        lna_new, b_new = dp.price_index(coef, np.log(price_1), number_goods, model = "QUAIDS")
        utility_new = ( ( (  obs[-1] - lna_new ) / b_new ) ** (-1) + lambda_new ) ** (-1)
        lambda_old = (lambd*np.log(price_0)).sum()
        lna_old, b_old = dp.price_index(coef, np.log(price_0), number_goods, model = "QUAIDS")
        lexp = (utility_new**(-1) - lambda_old)**(-1) * b_old + lna_old
        ev_util = np.exp(obs[-1]) - np.exp(lexp)
        return(ev_util)
        
ev_quaids_util  = ev_util(quaids_coef, price_1, price_0, number_goods, model = 'QUAIDS')
ev_aids_util  = ev_util(aids_coef, price_1, price_0, number_goods, model = 'AIDS')