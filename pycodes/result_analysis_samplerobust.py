#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:46:35 2018

@author: seugurlu
"""

from scipy.integrate import odeint
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from joblib import Parallel, delayed
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
import scipy as sp
import demand_predict as dp

number_goods = 10
number_bootstrap = 200
#number_cores = 35

#Set R functions
demand_system = importr('demand.tools')
demand_system_predict = demand_system.demand_system_predict
demand_system_elasticity = demand_system.demand_elasticity

#Define functions, set arrays
def cost(y_test, y_hat):
    return( 
            np.square( np.subtract(y_test, y_hat) ).stack().mean() 
            )

#Set Analysis
block1_comparison_test_costs = 1
block2_plotting_demand_functions = 0
block3_estimating_welfare_impact = 0
block4_plotting_welfare_impact = 0
block5_comparison_test_costs_demographics = 0
block6_plotting_demand_functions_demographics = 0
block7_elasticities = 0
block8_table_elasticities = 0
block9_cfnn_neg = 0
block10_plotting_demand_functions_neg = 0
block11_comparison_test_costs_robust = 0
block12_ls_ll_ratio = 0
block13_error_terms = 0
block14_income_elasticities = 0
#%%
'''Result 1 - Standard Estimation - Comparison of Test Costs'''
if block1_comparison_test_costs:
    sess = tf.InteractiveSession()
    cost_frame = pd.DataFrame(columns = ['AIDS', 'QUAIDS', 'ADN', 'TDN'])
    
    for sample_key in np.arange(number_bootstrap)+1:    
        #Import Data
        data = pd.HDFStore(r'data/processed/sample_robustness_data/sample'+str(sample_key), 'r')
        x_test = data['x_test']
        y_test = data['y_test']
        
        #Adjust data for passing to R
        log_price = np.log( x_test.filter(regex = r'^p.', axis = 1) )
        log_expenditure = np.log( x_test['total.expenditure'] )
        log_inputs = log_price.join(log_expenditure)
        
        log_price_r = pandas2ri.py2ri(log_price)
        log_expenditure_r = pandas2ri.py2ri(log_expenditure)
        y_r = pandas2ri.py2ri(y_test)
        
        #Import coefficients
        aids_coef = np.load(r"./output/robustness/sample splitting/aids/aids"+str(sample_key)+".npy", encoding = 'bytes')[0]
        y_hat_aids = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(aids_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )
        y_hat_aids.index = log_expenditure.index
        quaids_coef = np.load(r"./output/robustness/sample splitting/quaids/quaids"+str(sample_key)+".npy", encoding = 'bytes')[0]
        y_hat_quaids = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(quaids_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )
        y_hat_quaids.index = log_expenditure.index
        with np.load(r'./output/robustness/sample splitting/adn/adn'+str(sample_key)+'.npz', 'r') as adn:
            adn = adn['test_accuracy']
        with np.load(r'./output/robustness/sample splitting/tdn/tdn'+str(sample_key)+'.npz', 'r') as tdn:
            tdn = tdn['test_accuracy']
        
        cost_frame.loc[sample_key, 'AIDS'] = cost(y_test, y_hat_aids)   
        cost_frame.loc[sample_key, 'QUAIDS'] = cost(y_test, y_hat_quaids)   
        cost_frame.loc[sample_key, 'ADN'] = float( adn )
        cost_frame.loc[sample_key, 'TDN'] = float( tdn)
        data.close()
        
    #Set sns for graphs with a common style
    sns.set()
    sns.set_style("white")
    sns.set_context("paper")
    #1. Violin Plot of Cost Values
    fig = sns.violinplot(data = cost_frame, bw = 'silverman', inner = 'box', orient = 'h')
    sns.despine(left = True)
    fig.set_xlabel( r'$C( \mathbf{\Theta}, \mathcal{X}_t | \mathcal{X}_e, \mathcal{X}_c)$')
    plt.savefig('./output/robustness/sample splitting/figure/violin_standard_sample_robust.pdf', close = True)
        
    #Kolmogorov-Smirnov Tests
    #print('NFNN-CFNN KS:', sp.stats.ks_2samp(cost_frame['AN'], cost_frame['TN']))
    #print('AIDS-QUAIDS KS:', sp.stats.ks_2samp(cost_frame['AIDS'], cost_frame['QUAIDS']))
    #print('AIDS-NFNN KS:', sp.stats.ks_2samp(cost_frame['AIDS'], cost_frame['AN']))
    #print('AIDS-CFNN KS:', sp.stats.ks_2samp(cost_frame['AIDS'], cost_frame['TN']))
    #print('QUAIDS-NFNN KS:', sp.stats.ks_2samp(cost_frame['QUAIDS'], cost_frame['AN']))
    #print('QUAIDS-CFNN KS:', sp.stats.ks_2samp(cost_frame['QUAIDS'], cost_frame['TN']))
    
    sess.close()
#%%
if block2_plotting_demand_functions:
    sess = tf.InteractiveSession()
    '''Result 1 - Standard Estimation - Demand Functions'''
    #Initialize multiindex frame to hold predictions
    #Load Main Data Set
    main_data = pd.HDFStore(r'data/processed/main_sample', 'r')['main_sample']
    number_observation = main_data.shape[0]
    
    #Work with observed prices
    #log_price = np.log( main_data.filter(regex = r'^p.', axis = 1))
    #log_expenditure = np.log( main_data['total.expenditure'] )
    #log_inputs = log_price.join(log_expenditure).astype(np.float32)
    
    price = main_data.filter(regex = r'^p.', axis = 1)
    expenditure = pd.DataFrame(main_data['total.expenditure'])
    price_data = pd.DataFrame(price.median()).T.iloc[np.full(number_observation,0)].reset_index().drop('index', axis = 1)
    estimation_data = pd.concat([price_data,expenditure], axis = 1)
    log_inputs = np.log(estimation_data).astype(np.float32)
    
    cfnn_inputs = np.log(nf.impose_homogeneity(np.exp(log_inputs))).astype(np.float32)
    log_price_r = pandas2ri.py2ri(log_inputs.filter(regex = r'^p.', axis = 1))
    
    y = main_data.filter(regex = r'^w.', axis = 1)
    log_expenditure_r = pandas2ri.py2ri(log_inputs['total.expenditure'])
    y_r = pandas2ri.py2ri(y)
    
    #Generate multiindex
    index = main_data.index
    budget = y.keys()
    multi_index = pd.MultiIndex.from_product(iterables = [index, budget], names=['index', 'budget'])
    aids_predictions = pd.DataFrame(index=multi_index, columns=np.arange(1,number_bootstrap+1))
    quaids_predictions = pd.DataFrame(index=multi_index, columns=np.arange(1,number_bootstrap+1))
    nfnn_predictions = pd.DataFrame(index=multi_index, columns=np.arange(1,number_bootstrap+1))
    cfnn_predictions = pd.DataFrame(index=multi_index, columns=np.arange(1,number_bootstrap+1))
    
    for sample_key in range(number_bootstrap):
        aids_coef = np.load(r"./output/AIDS/aids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        y_hat_aids = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(aids_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )
        y_hat_aids.columns = y.keys()
        aids_predictions.iloc[:,sample_key] = y_hat_aids.stack()
        
        quaids_coef = np.load(r"./output/QUAIDS/quaids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        y_hat_quaids = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(quaids_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )
        y_hat_quaids.columns = y.keys()
        quaids_predictions.iloc[:,sample_key] = y_hat_quaids.stack()
        with np.load(r'./output/NN/nfnn/pycoefficients_nfnn'+str(sample_key+1)+'.npz', 'r') as nfnn:
            w_1 = nfnn['w_1']
            w_2 = nfnn['w_2']
            b_1 = nfnn['b_1']
            b_2 = nfnn['b_2']
            y_hat_nfnn = nf.forwardprop(log_inputs, w_1, w_2, b_1, b_2)[0].eval()
            y_hat_nfnn = pd.DataFrame(y_hat_nfnn, columns=y.keys())
            nfnn_predictions.iloc[:,sample_key] = y_hat_nfnn.stack()
        with np.load(r'./output/NN/cfnn/pycoefficients_cfnn'+str(sample_key+1)+'.npz', 'r') as cfnn:
            w_1 = cfnn['w_1']
            w_2 = cfnn['w_2']
            b_1 = cfnn['b_1']
            b_2 = cfnn['b_2']
            y_hat_cfnn = nf.forwardprop(cfnn_inputs, w_1, w_2, b_1, b_2)[0].eval()
            y_hat_cfnn = pd.DataFrame(y_hat_cfnn, columns=y.keys())
            cfnn_predictions.iloc[:,sample_key] = y_hat_cfnn.stack()    
    
    aids_results = pd.DataFrame(index = multi_index, columns = ['b95', 'mid', 'u95'])
    aids_results.loc[:,'mid'] = aids_predictions.mean(axis = 1)
    aids_results.loc[:,'b95'] = np.expand_dims(pd.DataFrame(np.sort(aids_predictions)).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
    aids_results.loc[:,'u95'] = np.expand_dims(pd.DataFrame(np.sort(aids_predictions)).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
    
    quaids_results = pd.DataFrame(index = multi_index, columns = ['b95', 'mid', 'u95'])
    quaids_results.loc[:,'mid'] = quaids_predictions.mean(axis = 1)
    quaids_results.loc[:,'b95'] = np.expand_dims(pd.DataFrame(np.sort(quaids_predictions)).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
    quaids_results.loc[:,'u95'] = np.expand_dims(pd.DataFrame(np.sort(quaids_predictions)).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
    
    nfnn_results = pd.DataFrame(index = multi_index, columns = ['b95', 'mid', 'u95'])
    nfnn_results.loc[:,'mid'] = nfnn_predictions.mean(axis = 1)
    nfnn_results.loc[:,'b95'] = np.expand_dims(pd.DataFrame(np.sort(nfnn_predictions)).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
    nfnn_results.loc[:,'u95'] = np.expand_dims(pd.DataFrame(np.sort(nfnn_predictions)).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
    
    cfnn_results = pd.DataFrame(index = multi_index, columns = ['b95', 'mid', 'u95'])
    cfnn_results.loc[:,'mid'] = cfnn_predictions.mean(axis = 1)
    cfnn_results.loc[:,'b95'] = np.expand_dims(pd.DataFrame(np.sort(cfnn_predictions)).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
    cfnn_results.loc[:,'u95'] = np.expand_dims(pd.DataFrame(np.sort(cfnn_predictions)).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
    
    x_axis = np.log(main_data.loc[:,'total.expenditure']) #X-axis of all graphs
    label_list = {'wt.breadcereal': 'Bread and Cereals', 
                  'wt.meat' : 'Meat', 
                  'wt.fish' : 'Fish and Seafood', 
                  'wt.milk' : 'Milk, Cheese, and Eggs',
                  'wt.fat' : 'Oils and Fats', 
                  'wt.fruit' : 'Fruit',
                  'wt.vegetable' : 'Vegetables', 
                  'wt.sugar' : 'Sugar, Jam, Confectionery', 
                  'wt.other' : 'Other', 
                  'wt.nonalc' : 'Non-Alcoholic Beverages'} #Label list for y-axis labeling
    model_list = ['AIDS', 'QUAIDS', 'NFNN', 'CFNN']
        
    for w in y.keys():
        
        AIDS_predictions = aids_results.xs(w, level = 'budget')
        QUAIDS_predictions = quaids_results.xs(w, level = 'budget')
        NFNN_predictions = nfnn_results.xs(w, level = 'budget')
        CFNN_predictions = cfnn_results.xs(w, level = 'budget')
        y_axis = y.loc[:,w]
        y_label = label_list[w]
        
        plot_data = pd.DataFrame( columns = model_list + ['total_expenditure'] )
        plot_data['total_expenditure'] = log_expenditure
        for model in model_list:
            plot_data[model] = eval(model+'_predictions')['mid']
        plot_data['CFNN Upper Bound'] = CFNN_predictions['u95']
        plot_data['CFNN Lower Bound'] = CFNN_predictions['b95']
        
        sns.set()
        sns.set_style("white")
        sns.set_context("paper")
        _, ax = plt.subplots()
        ax.set(xlim = (log_expenditure.min()*0.95, log_expenditure.max()*1.05 ))
        sns.regplot(x = 'total_expenditure', y = 'AIDS', data = plot_data, 
                    ax = ax, scatter = False, label = 'AIDS', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'QUAIDS', data = plot_data, 
                    ax = ax, scatter = False, label = 'QUAIDS', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'NFNN', data = plot_data, 
                   ax = ax, scatter = False, label = 'ADN', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN', data = plot_data, 
                   ax = ax, scatter = False, label = 'TDN', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN Upper Bound', data = plot_data, 
                   ax = ax, scatter = False, label = 'TDN 95% CI', lowess= True, color = 'black', line_kws= {'ls':'dashed', 'lw': 1})
        sns.regplot(x = 'total_expenditure', y = 'CFNN Lower Bound', data = plot_data, 
                   ax = ax, scatter = False, lowess= True, color = 'black', line_kws= {'ls': 'dashed', 'lw': 1})
        sns.despine()
        ax.set(xlabel='Log of Total Food Expenditure', ylabel = y_label)
        plt.legend()
        plt.savefig('./output/figure/'+w+'.pdf', close = True)
        sess.close()

#%%
if block3_estimating_welfare_impact:
    def welfare_graph(sample_key):
        main_data = pd.HDFStore(r'data/processed/main_sample', 'r')['main_sample']
        
        log_price = np.log( main_data.filter(regex = r'^p.', axis = 1) )
        log_expenditure = np.log( main_data['total.expenditure'] )
        log_inputs = log_price.join(log_expenditure).astype(np.float32)
        #log_inputs = log_inputs.sample(n = 200, random_state = 100)
                
        def price(t):
            return((price_1-price_0)*t + price_0)
        
        #Initialize Arrays
        index = log_inputs.index
        approach = ['AIDS', 'QUAIDS', 'CFNN']
        multi_index = pd.MultiIndex.from_product([index,approach], names = ['index', 'approach'])
        equiv_var = pd.DataFrame(index = multi_index, columns = np.arange(1,number_goods+1))
        
        multi_index = pd.MultiIndex.from_product([index, approach[:-1]], names = ['index', 'apprach'])
        util_var = pd.DataFrame(index = multi_index, columns = np.arange(1,number_goods+1))
        
        #Import coefficients
        aids_coef = np.array(np.load(r"./output/AIDS/aids"+str(sample_key)+".npy", encoding = 'bytes')[0])
        quaids_coef = np.array(np.load(r"./output/QUAIDS/quaids"+str(sample_key)+".npy", encoding = 'bytes')[0])
        with np.load(r'./output/NN/cfnn/pycoefficients_cfnn'+str(sample_key)+'.npz', 'r') as cfnn:
                w_1 = cfnn['w_1']
                w_2 = cfnn['w_2']
                b_1 = cfnn['b_1']
                b_2 = cfnn['b_2']
                
        for good in np.arange(number_goods):
            #Define price event
            price_change_good = good
            change_percent = 1.10
            for household_no in range(log_inputs.shape[0]) :
                household_index = log_inputs.index[household_no]
                household = np.array(log_inputs[household_no:household_no+1]).ravel()
                price_0 = np.exp(household[:-1])
                price_change = np.repeat(1.,number_goods)
                price_change[price_change_good] = change_percent
                price_1 = price_change * price_0
                
                def ev_util(coef, price_1, price_0, number_goods, model = 'AIDS'):
                    if model == 'AIDS':
                        lna_new, b_new = dp.price_index(coef, np.log(price_1), number_goods, model = "AIDS")
                        utility_new = (  household[-1] - lna_new ) / b_new
                        lna_old, b_old = dp.price_index(aids_coef, np.log(price_0), number_goods, model = "AIDS")
                        lexp = utility_new * b_old + lna_old
                        ev_util = np.exp(household[-1]) - np.exp(lexp)
                        
                    if model == 'QUAIDS':
                        _,_,_,lambd = dp.unroll_coef(coef, number_goods, model = 'QUAIDS')
                        lambda_new = (lambd*np.log(price_1)).sum()
                        lna_new, b_new = dp.price_index(coef, np.log(price_1), number_goods, model = "QUAIDS")
                        utility_new = ( ( (  household[-1] - lna_new ) / b_new ) ** (-1) + lambda_new ) ** (-1)
                        lambda_old = (lambd*np.log(price_0)).sum()
                        lna_old, b_old = dp.price_index(coef, np.log(price_0), number_goods, model = "QUAIDS")
                        lexp = (utility_new**(-1) - lambda_old)**(-1) * b_old + lna_old
                        ev_util = np.exp(household[-1]) - np.exp(lexp)
                    return(ev_util)
                        
                ev_aids_util = ev_util(aids_coef, price_1, price_0, number_goods, model = 'AIDS')
                ev_quaids_util = ev_util(quaids_coef, price_1, price_0, number_goods, model = 'QUAIDS')
                
                def f_aids(y,t):
                    func_input = np.append( price(t), np.exp(household[-1]) - y ) 
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
                    
                def f_quaids(y,t):
                    func_input = np.append( price(t), np.exp(household[-1]) - y ) 
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
        
                def f_cfnn(y,t):
                    func_input = np.append( price(t), np.exp(household[-1]) - y ) 
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
                
                t = np.linspace(1,0,20)
                ev_aids = odeint(f_aids, y0, t, rtol = 1e-4, atol = 1e-4)[-1][0]
                ev_quaids = odeint(f_quaids, y0, t, rtol = 1e-4, atol = 1e-4)[-1][0]
                ev_cfnn = odeint(f_cfnn, y0, t, rtol = 1e-4, atol = 1e-4)[-1][0]
                    
                equiv_var.loc[(household_index,'AIDS'),good+1] = ev_aids
                equiv_var.loc[(household_index,'QUAIDS'),good+1] = ev_quaids
                equiv_var.loc[(household_index,'CFNN'),good+1] = ev_cfnn
                util_var.loc[(household_index,'AIDS'),good+1] = ev_aids_util
                util_var.loc[(household_index,'QUAIDS'),good+1] = ev_quaids_util
                        
        equiv_var.to_csv('./output/other/ev_sample'+str(sample_key)+'.tsv', sep='\t')
        equiv_var.to_csv('./output/other/util_sample'+str(sample_key)+'.tsv', sep = '\t' )
            
    Parallel(n_jobs = number_cores)(delayed(welfare_graph)(sample_key+1) for sample_key in range(number_bootstrap))
#%%
#'''Plot Welfare Impact'''
if block4_plotting_welfare_impact:
    main_data = pd.HDFStore(r'data/processed/main_sample', 'r')['main_sample']        
    log_price = np.log( main_data.filter(regex = r'^p.', axis = 1) )
    log_expenditure = np.log( main_data['total.expenditure'] )
    log_inputs = log_price.join(log_expenditure).astype(np.float32)
   # log_inputs = log_inputs.sample(n = 200, random_state = 100)
    y = main_data.filter(regex = r'^w.', axis = 1)
    
    index = log_inputs.index
    approach = ['AIDS', 'QUAIDS', 'CFNN']
    multi_index = pd.MultiIndex.from_product([index,approach], names = ['index', 'approach'])
    multi_index_column = pd.MultiIndex.from_product([np.arange(number_bootstrap)+1, np.arange(number_goods)+1],
                                                    names = ['sample_key', 'good'])
    equiv_var = pd.DataFrame(index = multi_index, columns =multi_index_column)
      
    idx = pd.IndexSlice
    for sample in np.arange(number_bootstrap)+1:
       equiv_var_temp = pd.read_csv(r'./output/other/ev_sample'+str(sample)+'.tsv', sep = '\t', index_col = [0,1])
       equiv_var_temp.columns = pd.MultiIndex.from_product([[sample], np.arange(number_goods)+1],
                                                    names = ['sample_key', 'good'])
       equiv_var.loc[:,idx[sample,:]] = equiv_var_temp
    
    equiv_var_sort = equiv_var.sort_index()
    
    multi_column = pd.MultiIndex.from_product([approach,['b95', 'mid', 'u95']], names = ['approach', 'estimates'])
    label_list = {'wt.breadcereal': 'Bread and Cereals', 
                  'wt.meat' : 'Meat', 
                  'wt.fish' : 'Fish and Seafood', 
                  'wt.milk' : 'Milk, Cheese, and Eggs',
                  'wt.fat' : 'Oils and Fats', 
                  'wt.fruit' : 'Fruit',
                  'wt.vegetable' : 'Vegetables', 
                  'wt.sugar' : 'Sugar, Jam, Confectionery', 
                  'wt.other' : 'Other', 
                  'wt.nonalc' : 'Non-Alcoholic Beverages'} #Label list for y-axis labeling
                  
    for good in range(number_goods):
        col_name = y.columns[good]
        good_name = label_list[col_name]
        welfare_results = pd.DataFrame(index = index, columns = multi_column)
        welfare_results['total_expenditure'] = log_inputs['total.expenditure']
        welfare_results = welfare_results.sort_index()
        welfare_results = welfare_results.sort_index(axis = 1)
        
        for model in approach:
            welfare_results.loc[:,idx[model,'mid']] =  np.expand_dims(equiv_var_sort.loc[idx[:,model],idx[:,good+1]].mean(axis=1), axis =1)
            welfare_results.loc[:,idx[model,'b95']] = np.expand_dims(pd.DataFrame(np.sort(equiv_var_sort.loc[idx[:,model],idx[:,good+1]])).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
            welfare_results.loc[:,idx[model,'u95']] = np.expand_dims(pd.DataFrame(np.sort(equiv_var_sort.loc[idx[:,model],idx[:,good+1]])).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
        
        aids_results = welfare_results.loc[:, idx['AIDS']]
        quaids_results = welfare_results.loc[:, idx['QUAIDS']]
        cfnn_results = welfare_results.loc[:, idx['CFNN']]
        
        plot_data = pd.DataFrame(columns = ['AIDS', 'QUAIDS', 'CFNN', 'CFNN Upper Bound', 'CFNN Lower Bound'])
        plot_data['AIDS'] = 100*aids_results['mid']/np.exp(welfare_results.loc[:,idx['total_expenditure']])
        plot_data['QUAIDS'] = 100*quaids_results['mid']/np.exp(welfare_results.loc[:,idx['total_expenditure']])
        plot_data['CFNN'] = 100*cfnn_results['mid']/np.exp(welfare_results.loc[:,idx['total_expenditure']])
        plot_data['CFNN Upper Bound'] = 100*cfnn_results['u95']/np.exp(welfare_results.loc[:,idx['total_expenditure']])
        plot_data['CFNN Lower Bound'] = 100*cfnn_results['b95']/np.exp(welfare_results.loc[:,idx['total_expenditure']])
        plot_data['total_expenditure'] = np.exp(welfare_results.loc[:,idx['total_expenditure']])
        
        sns.set()
        sns.set_style("white")
        sns.set_context("paper")
        _, ax = plt.subplots()
        ax.set(xlim = (plot_data['total_expenditure'].min()*0.95, plot_data['total_expenditure'].max()*1.05 ))
        sns.regplot(x = 'total_expenditure', y = 'AIDS', data = plot_data, 
                    ax = ax, scatter = False, label = 'AIDS', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'QUAIDS', data = plot_data, 
                    ax = ax, scatter = False, label = 'QUAIDS', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN', data = plot_data, 
                   ax = ax, scatter = False, label = 'TDN', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN Upper Bound', data = plot_data, 
                   ax = ax, scatter = False, label = 'TDN 95% CI', lowess= True, color = 'black', line_kws= {'ls':'dashed', 'lw': 1})
        sns.regplot(x = 'total_expenditure', y = 'CFNN Lower Bound', data = plot_data, 
                   ax = ax, scatter = False, lowess= True, color = 'black', line_kws= {'ls': 'dashed', 'lw': 1})
        sns.despine()
        ax.set(xlabel='Total Food Expenditure', ylabel = 'EV as % of Total Food Expenditure')
        plt.legend()
        plt.savefig('./output/figure/'+good_name+'_tax.pdf', close = True)
        
#%%
'''Demographics'''
if block5_comparison_test_costs_demographics:
    sess = tf.InteractiveSession()
    cost_frame_dem = pd.DataFrame(columns = ['AIDS', 'QUAIDS', 'ADN', 'TDN'])
    for sample_key in range(number_bootstrap):
        #Import Data
        data = pd.HDFStore(r'data/processed/bootstrap_sample'+str(sample_key+1), 'r')
        x_test = data['x_test_dem']
        y_test = data['y_test']
        
        #Adjust data for passing to R
        log_price = np.log( x_test.filter(regex = r'^p.', axis = 1) )
        log_expenditure = np.log( x_test['total.expenditure'] )
        demographics = x_test.filter(regex = r'^d.', axis = 1)
        demographics = demographics.drop('d.hh.size', axis = 1)
        log_inputs = log_price.join(log_expenditure).join(demographics)
        
        """Normalize Demographics"""
        main_sample = pd.HDFStore("./data/processed/main_sample", 'r')['main_sample']
        main_sample_demographics = main_sample.filter(regex = "^d")
        main_sample_demographics = main_sample_demographics.drop('d.hh.size',1)
        mean_demographics = main_sample_demographics.mean()
        std_demographics = main_sample_demographics.std()
         
        x_test.iloc[:,number_goods+1:] = ( x_test.filter(regex = "^d") - mean_demographics ) / std_demographics
        
        log_price_r = pandas2ri.py2ri(log_price)
        log_expenditure_r = pandas2ri.py2ri(log_expenditure)
        demographics_r = pandas2ri.py2ri(demographics)
        y_r = pandas2ri.py2ri(y_test)
        
        #Import coefficients
        
        aids_coef = np.load(r"./output/AIDS/Demographics/aids_dem_"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        y_hat_aids = pd.DataFrame( pandas2ri.ri2py( 
                demand_system_predict(aids_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS", 
                                      dem = demographics_r) ) )
        y_hat_aids.index = log_expenditure.index
        quaids_coef = np.load(r"./output/QUAIDS/Demographics/quaids_dem_"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        y_hat_quaids = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(quaids_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS", dem = demographics_r) ) )
        y_hat_quaids.index = log_expenditure.index
        with np.load(r'./output/NN/nfnn_dem/pycoefficients_nfnn_dem_'+str(sample_key+1)+'.npz', 'r') as nfnn:
            nfnn = nfnn['test_accuracy']
        with np.load(r'./output/NN/cfnn_dem/pycoefficients_cfnn_dem'+str(sample_key+1)+'.npz', 'r') as cfnn:
            cfnn = cfnn['test_accuracy']
        
        cost_frame_dem.loc[sample_key, 'AIDS'] = cost(y_test, y_hat_aids)   
        cost_frame_dem.loc[sample_key, 'QUAIDS'] = cost(y_test, y_hat_quaids)   
        cost_frame_dem.loc[sample_key, 'ADN'] = float( nfnn )
        cost_frame_dem.loc[sample_key, 'TDN'] = float( cfnn)
        data.close()
        
    #Set sns for graphs with a common style
    sns.set()
    sns.set_style("white")
    sns.set_context("paper")
    #1. Violin Plot of Cost Values
    fig = sns.violinplot(data = cost_frame_dem, bw = 'silverman', orient = 'h')
    sns.despine(left = True)
    fig.set_xlabel( r'$C( \mathbf{\Theta}, \mathcal{X}_t | \mathcal{X}_e, \mathcal{X}_c)$')
    plt.savefig('./output/figure/violin_standard_demographics.pdf', close = True)
        
    #Kolmogorov-Smirnov Tests
    #print('NFNN-CFNN KS:', sp.stats.ks_2samp(cost_frame_dem['AN'], cost_frame_dem['TN']))
    #print('AIDS-QUAIDS KS:', sp.stats.ks_2samp(cost_frame_dem['AIDS'], cost_frame_dem['QUAIDS']))
    #print('AIDS-NFNN KS:', sp.stats.ks_2samp(cost_frame_dem['AIDS'], cost_frame_dem['AN']))
    #print('AIDS-CFNN KS:', sp.stats.ks_2samp(cost_frame_dem['AIDS'], cost_frame_dem['TN']))
    #print('QUAIDS-NFNN KS:', sp.stats.ks_2samp(cost_frame_dem['QUAIDS'], cost_frame_dem['AN']))
    #print('QUAIDS-CFNN KS:', sp.stats.ks_2samp(cost_frame_dem['QUAIDS'], cost_frame_dem['TN']))
    
    sess.close()
#%%
if block6_plotting_demand_functions_demographics:
    sess = tf.InteractiveSession()
    #Load Main Data Set
    main_data = pd.HDFStore(r'data/processed/main_sample', 'r')['main_sample']
    
    #Adjust data for passing it to R
    log_price = np.log( main_data.filter(regex = r'^p.', axis = 1) )
    log_expenditure = np.log( main_data['total.expenditure'] )
    demographics = main_data.filter(regex = r'^d.', axis = 1)
    demographics = demographics.drop('d.hh.size', axis = 1)
    log_inputs = log_price.join(log_expenditure).astype(np.float32)
    cfnn_inputs = np.log(nf.impose_homogeneity(np.exp(log_inputs)))
    cfnn_inputs = cfnn_inputs.join(demographics).astype(np.float32)
    log_inputs = log_inputs.join(demographics).astype(np.float32)
    y = main_data.filter(regex = r'^w.', axis = 1)
    
    """Normalize Demographics"""
    mean_demographics = demographics.mean()
    std_demographics = demographics.std()
    
    log_inputs.iloc[:,number_goods+1:] = ( log_inputs.filter(regex = "^d") - mean_demographics ) / std_demographics
    cfnn_inputs.iloc[:,number_goods+1:] = ( cfnn_inputs.filter(regex = "^d") - mean_demographics ) / std_demographics
    
    log_price_r = pandas2ri.py2ri(log_price)
    log_expenditure_r = pandas2ri.py2ri(log_expenditure)
    demographics_r = pandas2ri.py2ri(demographics)
    y_r = pandas2ri.py2ri(y)
    
    #Generate multiindex
    index = main_data.index
    budget = y.keys()
    multi_index = pd.MultiIndex.from_product(iterables = [index, budget], names=['index', 'budget'])
    aids_predictions = pd.DataFrame(index=multi_index, columns=np.arange(1,number_bootstrap+1))
    quaids_predictions = pd.DataFrame(index=multi_index, columns=np.arange(1,number_bootstrap+1))
    nfnn_predictions = pd.DataFrame(index=multi_index, columns=np.arange(1,number_bootstrap+1))
    cfnn_predictions = pd.DataFrame(index=multi_index, columns=np.arange(1,number_bootstrap+1))
    
    for sample_key in range(number_bootstrap):
        aids_coef = np.load(r"./output/AIDS/Demographics/aids_dem_"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        y_hat_aids = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(aids_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS", 
                                  dem = demographics_r) ) )
        y_hat_aids.columns = y.keys()
        aids_predictions.iloc[:,sample_key] = y_hat_aids.stack()
        
        quaids_coef = np.load(r"./output/QUAIDS/Demographics/quaids_dem_"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        y_hat_quaids = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(quaids_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS",
                                  dem = demographics_r) ) )
        y_hat_quaids.columns = y.keys()
        quaids_predictions.iloc[:,sample_key] = y_hat_quaids.stack()
        with np.load(r'./output/NN/nfnn_dem/pycoefficients_nfnn_dem_'+str(sample_key+1)+'.npz', 'r') as nfnn:
            w_1 = nfnn['w_1']
            w_2 = nfnn['w_2']
            b_1 = nfnn['b_1']
            b_2 = nfnn['b_2']
            y_hat_nfnn = nf.forwardprop(log_inputs, w_1, w_2, b_1, b_2)[0].eval()
            y_hat_nfnn = pd.DataFrame(y_hat_nfnn, columns=y.keys())
            nfnn_predictions.iloc[:,sample_key] = y_hat_nfnn.stack()
        with np.load(r'./output/NN/cfnn_dem/pycoefficients_cfnn_dem'+str(sample_key+1)+'.npz', 'r') as cfnn:
            w_1 = cfnn['w_1']
            w_2 = cfnn['w_2']
            b_1 = cfnn['b_1']
            b_2 = cfnn['b_2']
            y_hat_cfnn = nf.forwardprop(cfnn_inputs, w_1, w_2, b_1, b_2)[0].eval()
            y_hat_cfnn = pd.DataFrame(y_hat_cfnn, columns=y.keys())
            cfnn_predictions.iloc[:,sample_key] = y_hat_cfnn.stack()    
    
    aids_results = pd.DataFrame(index = multi_index, columns = ['b95', 'mid', 'u95'])
    aids_results.loc[:,'mid'] = aids_predictions.median(axis = 1)
    aids_results.loc[:,'b95'] = np.expand_dims(pd.DataFrame(np.sort(aids_predictions)).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
    aids_results.loc[:,'u95'] = np.expand_dims(pd.DataFrame(np.sort(aids_predictions)).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
    
    quaids_results = pd.DataFrame(index = multi_index, columns = ['b95', 'mid', 'u95'])
    quaids_results.loc[:,'mid'] = quaids_predictions.median(axis = 1)
    quaids_results.loc[:,'b95'] = np.expand_dims(pd.DataFrame(np.sort(quaids_predictions)).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
    quaids_results.loc[:,'u95'] = np.expand_dims(pd.DataFrame(np.sort(quaids_predictions)).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
    
    nfnn_results = pd.DataFrame(index = multi_index, columns = ['b95', 'mid', 'u95'])
    nfnn_results.loc[:,'mid'] = nfnn_predictions.median(axis = 1)
    nfnn_results.loc[:,'b95'] = np.expand_dims(pd.DataFrame(np.sort(nfnn_predictions)).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
    nfnn_results.loc[:,'u95'] = np.expand_dims(pd.DataFrame(np.sort(nfnn_predictions)).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
    
    cfnn_results = pd.DataFrame(index = multi_index, columns = ['b95', 'mid', 'u95'])
    cfnn_results.loc[:,'mid'] = cfnn_predictions.median(axis = 1)
    cfnn_results.loc[:,'b95'] = np.expand_dims(pd.DataFrame(np.sort(cfnn_predictions)).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
    cfnn_results.loc[:,'u95'] = np.expand_dims(pd.DataFrame(np.sort(cfnn_predictions)).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
    
    x_axis = np.log(main_data.loc[:,'total.expenditure']) #X-axis of all graphs
    label_list = {'wt.breadcereal': 'Bread and Cereals', 
                  'wt.meat' : 'Meat', 
                  'wt.fish' : 'Fish and Seafood', 
                  'wt.milk' : 'Milk, Cheese, and Eggs',
                  'wt.fat' : 'Oils and Fats', 
                  'wt.fruit' : 'Fruit',
                  'wt.vegetable' : 'Vegetables', 
                  'wt.sugar' : 'Sugar, Jam, Confectionery', 
                  'wt.other' : 'Other', 
                  'wt.nonalc' : 'Non-Alcoholic Beverages'} #Label list for y-axis labeling
        
    model_list = ['AIDS', 'QUAIDS', 'NFNN', 'CFNN']
        
    for w in y.keys():
        
        AIDS_predictions = aids_results.xs(w, level = 'budget')
        QUAIDS_predictions = quaids_results.xs(w, level = 'budget')
        NFNN_predictions = nfnn_results.xs(w, level = 'budget')
        CFNN_predictions = cfnn_results.xs(w, level = 'budget')
        y_axis = y.loc[:,w]
        y_label = label_list[w]
        
        plot_data = pd.DataFrame( columns = model_list + ['total_expenditure'] )
        plot_data['total_expenditure'] = log_expenditure
        for model in model_list:
            plot_data[model] = eval(model+'_predictions')['mid']
        plot_data['CFNN Upper Bound'] = CFNN_predictions['u95']
        plot_data['CFNN Lower Bound'] = CFNN_predictions['b95']
        
        sns.set()
        sns.set_style("white")
        sns.set_context("paper")
        _, ax = plt.subplots()
        ax.set(xlim = (log_expenditure.min()*0.95, log_expenditure.max()*1.05 ))
        sns.regplot(x = 'total_expenditure', y = 'AIDS', data = plot_data, 
                    ax = ax, scatter = False, label = 'AIDS', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'QUAIDS', data = plot_data, 
                    ax = ax, scatter = False, label = 'QUAIDS', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'NFNN', data = plot_data, 
                   ax = ax, scatter = False, label = 'ADN', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN', data = plot_data, 
                   ax = ax, scatter = False, label = 'TDN', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN Upper Bound', data = plot_data, 
                   ax = ax, scatter = False, label = 'TDN 95% CI', lowess= True, color = 'black', line_kws= {'ls':'dashed', 'lw': 1})
        sns.regplot(x = 'total_expenditure', y = 'CFNN Lower Bound', data = plot_data, 
                   ax = ax, scatter = False, lowess= True, color = 'black', line_kws= {'ls': 'dashed', 'lw': 1})
        sns.despine()
        ax.set(xlabel='Log of Total Food Expenditure', ylabel = y_label)
        plt.legend()
        plt.savefig('./output/figure/'+w+'_demographics.pdf', close = True)
        plt.close
        
    for w in y.keys()[0:6]: #1 page takes 6 graphs :) 
        
        AIDS_predictions = aids_results.xs(w, level = 'budget')
        QUAIDS_predictions = quaids_results.xs(w, level = 'budget')
        NFNN_predictions = nfnn_results.xs(w, level = 'budget')
        CFNN_predictions = cfnn_results.xs(w, level = 'budget')
        y_axis = y.loc[:,w]
        y_label = label_list[w]
        
        plot_data = pd.DataFrame( columns = model_list + ['total_expenditure'] )
        plot_data['total_expenditure'] = log_expenditure
        for model in model_list:
            plot_data[model] = eval(model+'_predictions')['mid']
        plot_data['CFNN Upper Bound'] = CFNN_predictions['u95']
        plot_data['CFNN Lower Bound'] = CFNN_predictions['b95']
        plot_data['d_n_child'] = main_data['d.n.child']
        
        condition_0 = plot_data['d_n_child'] == 0
        condition_1 = plot_data['d_n_child'] == 1
        condition_2 = plot_data['d_n_child'] == 2
        
        sns.set()
        sns.set_style("white")
        sns.set_context("paper")
        _, ax = plt.subplots()
        ax.set(xlim = (log_expenditure.min()*0.95, log_expenditure.max()*1.05 ))
        sns.regplot(x = 'total_expenditure', y = 'QUAIDS', data = plot_data[condition_0], 
                    ax = ax, lowess = True, scatter = False, label = '# of children = 0')
        sns.regplot(x = 'total_expenditure', y = 'QUAIDS', data = plot_data[condition_1], 
                    ax = ax, lowess = True, scatter = False, label = '# of children = 1')
        sns.regplot(x = 'total_expenditure', y = 'QUAIDS', data = plot_data[condition_2], 
                    ax = ax, lowess = True, scatter = False, label = '# of children = 2')
        sns.despine()
        ax.set(xlabel='Log of Total Food Expenditure', ylabel = y_label)
        plt.legend()
        plt.savefig('./output/figure/'+w+'_demographics_quaidschild.pdf', close = True)
        plt.close
        
        sns.set()
        sns.set_style("white")
        sns.set_context("paper")
        _, ax = plt.subplots()
        
        ax.set(xlim = (log_expenditure.min()*0.95, log_expenditure.max()*1.05 ))
        sns.regplot(x = 'total_expenditure', y = 'CFNN', data = plot_data[condition_0], 
                    ax = ax, lowess = True, scatter = False, label = '# of children = 0')
        sns.regplot(x = 'total_expenditure', y = 'CFNN', data = plot_data[condition_1], 
                    ax = ax, lowess = True, scatter = False, label = '# of children = 1')
        sns.regplot(x = 'total_expenditure', y = 'CFNN', data = plot_data[condition_2], 
                    ax = ax, lowess = True, scatter = False, label = '# of children = 2')
        sns.despine()
        ax.set(xlabel='Log of Total Food Expenditure', ylabel = y_label)
        plt.legend()
        plt.savefig('./output/figure/'+w+'_demographics_cfnnchild.pdf', close = True)
        plt.close

        sess.close()
#%%
if block7_elasticities:
    sess = tf.InteractiveSession()
    #Load Main Data Set
    main_data = pd.HDFStore(r'data/processed/main_sample', 'r')['main_sample']
    
    #Adjust data for passing it to R
    price = main_data.filter(regex = r'^p.', axis = 1)
    expenditure = main_data['total.expenditure']
    demographics = main_data.filter(regex = r'^d.', axis = 1)
    demographics = demographics.drop('d.hh.size', axis = 1)
    elasticity_observation = pd.DataFrame(price.median()).T
    elasticity_observation['total.expenditure'] = expenditure.median()
    
    #Initialize DataFrame
    good = price.keys()
    model = ['AIDS', 'QUAIDS', 'NFNN', 'CFNN']
    multi_index_row = pd.MultiIndex.from_product([good, model], names = ['good', 'model'])
    multi_index_column = pd.MultiIndex.from_product([np.arange(number_bootstrap)+1, good], names = ['sample', 'good'])
    income_elasticity = pd.DataFrame(index = multi_index_row, columns = np.arange(number_bootstrap)+1 ).sort_index()
    uncompensated_price_elasticity = pd.DataFrame(index = multi_index_row, columns = multi_index_column ).sort_index()
    uncompensated_price_elasticity = uncompensated_price_elasticity.sort_index(1)
    compensated_price_elasticity = pd.DataFrame(index = multi_index_row, columns = multi_index_column ).sort_index()
    compensated_price_elasticity = compensated_price_elasticity.sort_index(1)
    slutsky_matrix = pd.DataFrame(index = multi_index_row, columns = multi_index_column ).sort_index()
    slutsky_matrix = slutsky_matrix.sort_index(1)
    
    #Inputs For AIDS and QUAIDS
    log_price = np.log(elasticity_observation.filter(regex = r'^p.', axis = 1))
    log_expenditure = np.log(elasticity_observation['total.expenditure'])
    log_price_r = pandas2ri.py2ri(log_price)
    log_expenditure_r = pandas2ri.py2ri(log_expenditure)
    
    #Inputs for NFNN and CFNN
    nfnn_input = np.log(elasticity_observation)
    cfnn_input = np.log(nf.impose_homogeneity(elasticity_observation))
    
    for sample_key in np.arange(number_bootstrap):
        aids_coef = np.load(r"./output/AIDS/aids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        inc_elas, uncomp_elas, comp_elas, slutsky_mat = demand_system_elasticity(aids_coef, log_price_r, log_expenditure_r, model = "AIDS")
        income_elasticity.loc[ (slice(None), 'AIDS'), sample_key+1] = inc_elas = pandas2ri.ri2py(inc_elas)
        uncompensated_price_elasticity.loc[ (slice(None),'AIDS'), (sample_key+1,slice(None))] = pandas2ri.ri2py(uncomp_elas)
        compensated_price_elasticity.loc[ (slice(None),'AIDS'), (sample_key+1,slice(None))] = pandas2ri.ri2py(comp_elas)
        slutsky_matrix.loc[ (slice(None),'AIDS'), (sample_key+1,slice(None))] = pandas2ri.ri2py(slutsky_mat)
        
        quaids_coef = np.load(r"./output/QUAIDS/quaids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        inc_elas, uncomp_elas, comp_elas, slutsky_mat = demand_system_elasticity(quaids_coef, log_price_r, log_expenditure_r, model = "QUAIDS")
        income_elasticity.loc[ (slice(None), 'QUAIDS'), sample_key+1] = pandas2ri.ri2py(inc_elas)
        uncompensated_price_elasticity.loc[ (slice(None),'QUAIDS'), (sample_key+1,slice(None))] = pandas2ri.ri2py(uncomp_elas)
        compensated_price_elasticity.loc[ (slice(None),'QUAIDS'), (sample_key+1,slice(None))] = pandas2ri.ri2py(comp_elas)
        slutsky_matrix.loc[ (slice(None),'QUAIDS'), (sample_key+1,slice(None))] = pandas2ri.ri2py(slutsky_mat)
        
        with np.load(r'./output/NN/nfnn/pycoefficients_nfnn'+str(sample_key+1)+'.npz', 'r') as nfnn:
            w_1 = nfnn['w_1']
            w_2 = nfnn['w_2']
            b_1 = nfnn['b_1']
            b_2 = nfnn['b_2']
            yhat, z1 = nf.neural_predict(nfnn_input,w_1,w_2,b_1,b_2)
            inc_elas, uncomp_elas, comp_elas, slutsky_mat = nf.slutsky_matrix(nfnn_input, yhat, w_1, w_2, z1, 
                                                                              number_goods, 'NFNN')
            income_elasticity.loc[ (slice(None), 'NFNN'), sample_key+1] = inc_elas
            uncompensated_price_elasticity.loc[ (slice(None),'NFNN'), (sample_key+1,slice(None))] = uncomp_elas
            compensated_price_elasticity.loc[ (slice(None),'NFNN'), (sample_key+1,slice(None))] = comp_elas
            slutsky_matrix.loc[ (slice(None),'NFNN'), (sample_key+1,slice(None))] = slutsky_mat
                
        with np.load(r'./output/NN/cfnn/pycoefficients_cfnn'+str(sample_key+1)+'.npz', 'r') as cfnn:
            w_1 = cfnn['w_1']
            w_2 = cfnn['w_2']
            b_1 = cfnn['b_1']
            b_2 = cfnn['b_2']
            yhat, z1 = nf.neural_predict(cfnn_input,w_1,w_2,b_1,b_2)
            inc_elas, uncomp_elas, comp_elas, slutsky_mat = nf.slutsky_matrix(cfnn_input, yhat, w_1, w_2, z1, 
                                                                              number_goods, 'CFNN')
            income_elasticity.loc[ (slice(None), 'CFNN'), sample_key+1] = inc_elas
            uncompensated_price_elasticity.loc[ (slice(None),'CFNN'), (sample_key+1,slice(None))] = uncomp_elas
            compensated_price_elasticity.loc[ (slice(None),'CFNN'), (sample_key+1,slice(None))] = comp_elas
            slutsky_matrix.loc[ (slice(None),'CFNN'), (sample_key+1,slice(None))] = slutsky_mat
            
    income_elasticity.to_pickle("./output/other/income_elasticity")
    uncompensated_price_elasticity.to_pickle("./output/other/uncompensated_price_elasticity")
    compensated_price_elasticity.to_pickle("./output/other/compensated_price_elasticity")
    slutsky_matrix.to_pickle("./output/other/slutsky_matrix")
    
    sess.close()
#%%Generate Elasticties with Bounds
if block8_table_elasticities:
    income_elasticity = pd.read_pickle("./output/other/income_elasticity")    
    uncompensated_price_elasticity = pd.read_pickle("./output/other/uncompensated_price_elasticity")
    compensated_price_elasticity = pd.read_pickle("./output/other/compensated_price_elasticity")
    slutsky_matrix = pd.read_pickle("./output/other/slutsky_matrix")
    
    #Income Elasticity
    aids_estimates = income_elasticity.xs('AIDS', level = 'model')
    quaids_estimates = income_elasticity.xs('QUAIDS', level = 'model')
    nfnn_estimates = income_elasticity.xs('NFNN', level = 'model')
    cfnn_estimates = income_elasticity.xs('CFNN', level = 'model')
    
    multi_index = pd.MultiIndex.from_product([income_elasticity.index.levels[0], np.arange(number_bootstrap)+1], names=['Good', 'Sample'])
    income_elas_frame = pd.DataFrame(index = multi_index, columns = ['AIDS', 'QUAIDS', 'ADN', 'TDN'])
    for good in income_elasticity.index.levels[0]:
        income_elas_frame.xs(good, level = 'Good').loc[:,'AIDS'] = aids_estimates.loc[good,:]
        income_elas_frame.xs(good, level = 'Good').loc[:,'QUAIDS'] = quaids_estimates.loc[good,:]
        income_elas_frame.xs(good, level = 'Good').loc[:,'ADN'] = nfnn_estimates.loc[good,:]
        income_elas_frame.xs(good, level = 'Good').loc[:,'TDN'] = cfnn_estimates.loc[good,:]
    
    sns.set()
    sns.set_style("white")
    sns.set_context("paper")
    label_list = {'pt.breadcereal': 'Bread and Cereals', 
                  'pt.meat' : 'Meat', 
                  'pt.fish' : 'Fish and Seafood', 
                  'pt.milk' : 'Milk, Cheese, and Eggs',
                  'pt.fat' : 'Oils and Fats', 
                  'pt.fruit' : 'Fruit',
                  'pt.vegetable' : 'Vegetables', 
                  'pt.sugar' : 'Sugar, Jam, Confectionery', 
                  'pt.other' : 'Other', 
                  'pt.nonalc' : 'Non-Alcoholic Beverages'} #Label list for y-axis labeling
    goods = income_elasticity.index.levels[0]
    for good in goods:
        fig = sns.boxplot ( data = income_elas_frame.xs(good, level = 'Good'), orient = 'h')
        sns.despine(left = True)
        fig.set_xlabel( r'Income Elasticity of '+ label_list[good])
        plt.savefig('./output/figure/violin_income_elas_'+good+'.pdf', close = True)
        plt.close()
    
    #Mean Income Elasticities PairPlot
    income_elas_frame_mean = pd.DataFrame(columns = ['AIDS', 'QUAIDS', 'ADN', 'TDN'], index = goods)
    for good in goods:
        income_elas_frame_mean.loc[good,:] = income_elas_frame.loc[good,:].apply(np.mean)
    
    estimates = ['AIDS','QUAIDS','ADN','TDN']
    
    fig, axes = plt.subplots(ncols=4, nrows=4)
    for estimator in np.arange(4):
        for estimator2 in np.arange(estimator+1,4):
            est_col = estimates[estimator]
            est_row = estimates[estimator2]
            pp = sns.regplot(data = income_elas_frame_mean, y = est_row, x = est_col, fit_reg = False, ax = axes[estimator2][estimator])
            pp.plot([income_elas_frame_mean.loc[:,[est_row,est_col]].values.min(),
                      income_elas_frame_mean.loc[:,[est_row,est_col]].values.max()],
                    [income_elas_frame_mean.loc[:,[est_row,est_col]].values.min(),
                     income_elas_frame_mean.loc[:,[est_row,est_col]].values.max()])
    for i in np.arange(4):
        for j in np.arange(4):
            if i != 3:
                axes[i][j].set_xlabel("")
                axes[i][j].get_xaxis().set_ticks([])
            if j != 0 :
                axes[i][j].set_ylabel("")
                axes[i][j].get_yaxis().set_ticks([])
    for i in np.arange(4):
        axes[0][i].remove()    
    for i in np.arange(1,4):
        axes[1][i].remove()
    for i in np.arange(2,4):
        axes[2][i].remove()
    axes[3][3].remove()
    plt.savefig('./output/figure/income_elasticity_45comp.pdf', close = True)
    plt.close
    
    
    #Uncompensated Price Elasticity
    aids_estimates = uncompensated_price_elasticity.xs('AIDS', level = 'model')
    quaids_estimates = uncompensated_price_elasticity.xs('QUAIDS', level = 'model')
    nfnn_estimates = uncompensated_price_elasticity.xs('NFNN', level = 'model')
    cfnn_estimates = uncompensated_price_elasticity.xs('CFNN', level = 'model')
    
    rows = np.array(income_elasticity.index.levels[0])
    multi_index = pd.MultiIndex.from_product( [rows, ['b95', 'mid', 'u95']])
    aids_results = pd.DataFrame(index = rows, columns = multi_index)
    quaids_results = pd.DataFrame(index = rows, columns = multi_index)
    nfnn_results = pd.DataFrame(index = rows, columns = multi_index)
    cfnn_results = pd.DataFrame(index = rows, columns = multi_index)
    
    for good in income_elasticity.index.levels[0]:
        aids_results.loc[:,(good,'mid')] = aids_estimates.xs(good,axis = 1, level = 'good').mean(axis = 1)
        aids_results.loc[:,(good,'b95')] = np.sort(aids_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.05*number_bootstrap)-1]
        aids_results.loc[:,(good,'u95')] = np.sort(aids_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.95*number_bootstrap)-1]
        
        quaids_results.loc[:,(good,'mid')] = quaids_estimates.xs(good,axis = 1, level = 'good').mean(axis = 1)
        quaids_results.loc[:,(good,'b95')] = np.sort(quaids_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.05*number_bootstrap)-1]
        quaids_results.loc[:,(good,'u95')] = np.sort(quaids_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.95*number_bootstrap)-1]
        
        nfnn_results.loc[:,(good,'mid')] = nfnn_estimates.xs(good,axis = 1, level = 'good').mean(axis = 1)
        nfnn_results.loc[:,(good,'b95')] = np.sort(nfnn_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.05*number_bootstrap)-1]
        nfnn_results.loc[:,(good,'u95')] = np.sort(nfnn_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.95*number_bootstrap)-1]
        
        cfnn_results.loc[:,(good,'mid')] = cfnn_estimates.xs(good,axis = 1, level = 'good').mean(axis = 1)
        cfnn_results.loc[:,(good,'b95')] = np.sort(cfnn_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.05*number_bootstrap)-1]
        cfnn_results.loc[:,(good,'u95')] = np.sort(cfnn_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.95*number_bootstrap)-1]
        
        #Uncomp Price Elasticities PairPlot
        uncomp_price_mean = pd.DataFrame(columns = ['AIDS', 'QUAIDS', 'ADN', 'TDN'])
        uncomp_price_mean['AIDS'] = aids_results.xs('mid',level = 1, axis = 1, drop_level = True).values.flatten()
        uncomp_price_mean['QUAIDS'] = quaids_results.xs('mid',level = 1, axis = 1, drop_level = True).values.flatten()
        uncomp_price_mean['ADN'] = nfnn_results.xs('mid',level = 1, axis = 1, drop_level = True).values.flatten()
        uncomp_price_mean['TDN'] = cfnn_results.xs('mid',level = 1, axis = 1, drop_level = True).values.flatten()
        
        estimates = ['AIDS','QUAIDS','AN','TN']
            
        fig, axes = plt.subplots(ncols=4, nrows=4)
        for estimator in np.arange(4):
            for estimator2 in np.arange(estimator+1,4):
                est_col = estimates[estimator]
                est_row = estimates[estimator2]
                pp = sns.regplot(data = uncomp_price_mean, y = est_row, x = est_col, fit_reg = False, ax = axes[estimator2][estimator])
                pp.plot([uncomp_price_mean.loc[:,[est_row,est_col]].values.min(),
                         uncomp_price_mean.loc[:,[est_row,est_col]].values.max()],
                        [uncomp_price_mean.loc[:,[est_row,est_col]].values.min(),
                         uncomp_price_mean.loc[:,[est_row,est_col]].values.max()])
        for i in np.arange(4):
            for j in np.arange(4):
                if i != 3:
                    axes[i][j].set_xlabel("")
                    axes[i][j].get_xaxis().set_ticks([])
                if j != 0 :
                    axes[i][j].set_ylabel("")
                    axes[i][j].get_yaxis().set_ticks([])
            
        for i in np.arange(4):
           axes[0][i].remove()    
        for i in np.arange(1,4):
           axes[1][i].remove()
        for i in np.arange(2,4):
           axes[2][i].remove()
        axes[3][3].remove()
        plt.savefig('./output/figure/uncomp_price_elasticity_45comp.pdf', close = True)
        plt.close
            
    #Compensated Price Elasticity
    aids_estimates = compensated_price_elasticity.xs('AIDS', level = 'model')
    quaids_estimates = compensated_price_elasticity.xs('QUAIDS', level = 'model')
    nfnn_estimates = compensated_price_elasticity.xs('NFNN', level = 'model')
    cfnn_estimates = compensated_price_elasticity.xs('CFNN', level = 'model')
    
    rows = np.array(income_elasticity.index.levels[0])
    multi_index = pd.MultiIndex.from_product( [rows, ['b95', 'mid', 'u95']])
    aids_results = pd.DataFrame(index = rows, columns = multi_index)
    quaids_results = pd.DataFrame(index = rows, columns = multi_index)
    nfnn_results = pd.DataFrame(index = rows, columns = multi_index)
    cfnn_results = pd.DataFrame(index = rows, columns = multi_index)
    
    for good in income_elasticity.index.levels[0]:
        aids_results.loc[:,(good,'mid')] = aids_estimates.xs(good,axis = 1, level = 'good').mean(axis = 1)
        aids_results.loc[:,(good,'b95')] = np.sort(aids_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.05*number_bootstrap)-1]
        aids_results.loc[:,(good,'u95')] = np.sort(aids_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.95*number_bootstrap)-1]
        
        quaids_results.loc[:,(good,'mid')] = quaids_estimates.xs(good,axis = 1, level = 'good').mean(axis = 1)
        quaids_results.loc[:,(good,'b95')] = np.sort(quaids_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.05*number_bootstrap)-1]
        quaids_results.loc[:,(good,'u95')] = np.sort(quaids_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.95*number_bootstrap)-1]
        
        nfnn_results.loc[:,(good,'mid')] = nfnn_estimates.xs(good,axis = 1, level = 'good').mean(axis = 1)
        nfnn_results.loc[:,(good,'b95')] = np.sort(nfnn_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.05*number_bootstrap)-1]
        nfnn_results.loc[:,(good,'u95')] = np.sort(nfnn_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.95*number_bootstrap)-1]
        
        cfnn_results.loc[:,(good,'mid')] = cfnn_estimates.xs(good,axis = 1, level = 'good').mean(axis = 1)
        cfnn_results.loc[:,(good,'b95')] = np.sort(cfnn_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.05*number_bootstrap)-1]
        cfnn_results.loc[:,(good,'u95')] = np.sort(cfnn_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.95*number_bootstrap)-1]
        
    #Slutsky Matrix
    aids_estimates = slutsky_matrix.xs('AIDS', level = 'model')
    quaids_estimates = slutsky_matrix.xs('QUAIDS', level = 'model')
    nfnn_estimates = slutsky_matrix.xs('NFNN', level = 'model')
    cfnn_estimates = slutsky_matrix.xs('CFNN', level = 'model')
    
    rows = np.array(income_elasticity.index.levels[0])
    multi_index = pd.MultiIndex.from_product( [rows, ['b95', 'mid', 'u95']])
    aids_results = pd.DataFrame(index = rows, columns = multi_index)
    quaids_results = pd.DataFrame(index = rows, columns = multi_index)
    nfnn_results = pd.DataFrame(index = rows, columns = multi_index)
    cfnn_results = pd.DataFrame(index = rows, columns = multi_index)
    
    for good in income_elasticity.index.levels[0]:
        aids_results.loc[:,(good,'mid')] = aids_estimates.xs(good,axis = 1, level = 'good').mean(axis = 1)
        aids_results.loc[:,(good,'b95')] = np.sort(aids_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.05*number_bootstrap)-1]
        aids_results.loc[:,(good,'u95')] = np.sort(aids_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.95*number_bootstrap)-1]
        
        quaids_results.loc[:,(good,'mid')] = quaids_estimates.xs(good,axis = 1, level = 'good').mean(axis = 1)
        quaids_results.loc[:,(good,'b95')] = np.sort(quaids_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.05*number_bootstrap)-1]
        quaids_results.loc[:,(good,'u95')] = np.sort(quaids_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.95*number_bootstrap)-1]
        
        nfnn_results.loc[:,(good,'mid')] = nfnn_estimates.xs(good,axis = 1, level = 'good').mean(axis = 1)
        nfnn_results.loc[:,(good,'b95')] = np.sort(nfnn_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.05*number_bootstrap)-1]
        nfnn_results.loc[:,(good,'u95')] = np.sort(nfnn_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.95*number_bootstrap)-1]
        
        cfnn_results.loc[:,(good,'mid')] = cfnn_estimates.xs(good,axis = 1, level = 'good').mean(axis = 1)
        cfnn_results.loc[:,(good,'b95')] = np.sort(cfnn_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.05*number_bootstrap)-1]
        cfnn_results.loc[:,(good,'u95')] = np.sort(cfnn_estimates.xs(good,axis = 1, level = 'good'))[:,int(0.95*number_bootstrap)-1]
        
#%%
'''Plot Price Elasticities'''
#pair_data = pd.DataFrame(aids_results.xs('mid',level = 1, axis = 1, drop_level = True).values.flatten(), columns = ["AIDS"]) 
#pair_data['QUAIDS'] = pd.DataFrame(quaids_results.xs('mid',level = 1, axis = 1, drop_level = True).values.flatten() )
#pair_data['AN'] = pd.DataFrame(nfnn_results.xs('mid',level = 1, axis = 1, drop_level = True).values.flatten() )
#pair_data['TN'] = pd.DataFrame(cfnn_results.xs('mid',level = 1, axis = 1, drop_level = True).values.flatten() )

#g = sns.pairplot(pair_data)
#for i,j in zip(*np.triu_indices_from(g.axes,1)):
#g.axes[i,j].set_visible(False)

#g = sns.PairGrid(pair_data)
#g.map_diag(sns.distplot, kde = False)
#g.map_offdiag(sns.regplot, fit_reg = False)
#for i,j in zip(*np.triu_indices_from(g.axes,1)):
#g.axes[i,j].set_visible(False)
#g.axes.[i,j].plot([0,1],[0,1])

#%%
'''Result 1 - Standard Estimation - Comparison of Test Costs'''
if block9_cfnn_neg:
    sess = tf.InteractiveSession()
    cost_frame = pd.DataFrame(columns = ['AIDS', 'QUAIDS', 'ADN', 'TDN', 'TDN-N'])
    for sample_key in range(number_bootstrap):
        #Import Data
        data = pd.HDFStore(r'data/processed/bootstrap_sample'+str(sample_key+1), 'r')
        x_test = data['x_test']
        y_test = data['y_test']
        
        #Adjust data for passing to R
        log_price = np.log( x_test.filter(regex = r'^p.', axis = 1) )
        log_expenditure = np.log( x_test['total.expenditure'] )
        log_inputs = log_price.join(log_expenditure)
        
        log_price_r = pandas2ri.py2ri(log_price)
        log_expenditure_r = pandas2ri.py2ri(log_expenditure)
        y_r = pandas2ri.py2ri(y_test)
        
        #Import coefficients
        aids_coef = np.load(r"./output/AIDS/aids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        y_hat_aids = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(aids_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )
        y_hat_aids.index = log_expenditure.index
        quaids_coef = np.load(r"./output/QUAIDS/quaids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        y_hat_quaids = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(quaids_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )
        y_hat_quaids.index = log_expenditure.index
        with np.load(r'./output/NN/nfnn/pycoefficients_nfnn'+str(sample_key+1)+'.npz', 'r') as nfnn:
            nfnn = nfnn['test_accuracy']
        with np.load(r'./output/NN/cfnn/pycoefficients_cfnn'+str(sample_key+1)+'.npz', 'r') as cfnn:
            cfnn = cfnn['test_accuracy']
        with np.load(r'./output/NN/cfnn_neg/pycoefficients_cfnn_neg_'+str(sample_key+1)+'.npz', 'r') as cfnn_neg:
            cfnn_neg = cfnn_neg['test_accuracy']
        
        cost_frame.loc[sample_key, 'AIDS'] = cost(y_test, y_hat_aids)   
        cost_frame.loc[sample_key, 'QUAIDS'] = cost(y_test, y_hat_quaids)   
        cost_frame.loc[sample_key, 'ADN'] = float( nfnn )
        cost_frame.loc[sample_key, 'TDN'] = float( cfnn)
        cost_frame.loc[sample_key, 'TDN-N'] = float( cfnn_neg )
        data.close()
        
    #Set sns for graphs with a common style
    sns.set()
    sns.set_style("white")
    sns.set_context("paper")
    #1. Violin Plot of Cost Values
    fig = sns.violinplot(data = cost_frame, bw = 'silverman', inner = 'box', orient = 'h')
    sns.despine(left = True)
    fig.set_xlabel( r'$C(\Theta, \mathcal{X}_t | \mathcal{X}_e, \mathcal{X}_c)$')
    plt.savefig('./output/figure/negativity/violin_negativity.pdf', close = True)
        
    sess.close()

#%%
if block10_plotting_demand_functions_neg:
    sess = tf.InteractiveSession()
    '''Result 1 - Standard Estimation - Demand Functions'''
    #Initialize multiindex frame to hold predictions
    #Load Main Data Set
    main_data = pd.HDFStore(r'data/processed/main_sample', 'r')['main_sample']
    
    #Work with observed prices
    log_price = np.log( main_data.filter(regex = r'^p.', axis = 1))
    log_expenditure = np.log( main_data['total.expenditure'] )
    log_inputs = log_price.join(log_expenditure).astype(np.float32)
    cfnn_inputs = np.log(nf.impose_homogeneity(np.exp(log_inputs))).astype(np.float32)
    log_inputs = log_price.join(log_expenditure).astype(np.float32)
    log_price_r = pandas2ri.py2ri(log_price)
    
    y = main_data.filter(regex = r'^w.', axis = 1)
    log_expenditure_r = pandas2ri.py2ri(log_expenditure)
    y_r = pandas2ri.py2ri(y)
    
    #Generate multiindex
    index = main_data.index
    budget = y.keys()
    multi_index = pd.MultiIndex.from_product(iterables = [index, budget], names=['index', 'budget'])
    aids_predictions = pd.DataFrame(index=multi_index, columns=np.arange(1,number_bootstrap+1))
    quaids_predictions = pd.DataFrame(index=multi_index, columns=np.arange(1,number_bootstrap+1))
    nfnn_predictions = pd.DataFrame(index=multi_index, columns=np.arange(1,number_bootstrap+1))
    cfnn_predictions = pd.DataFrame(index=multi_index, columns=np.arange(1,number_bootstrap+1))
    cfnn_neg_predictions = pd.DataFrame(index=multi_index, columns=np.arange(1,number_bootstrap+1))
    
    for sample_key in range(number_bootstrap):
        aids_coef = np.load(r"./output/AIDS/aids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        y_hat_aids = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(aids_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )
        y_hat_aids.columns = y.keys()
        aids_predictions.iloc[:,sample_key] = y_hat_aids.stack()
        
        quaids_coef = np.load(r"./output/QUAIDS/quaids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        y_hat_quaids = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(quaids_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )
        y_hat_quaids.columns = y.keys()
        quaids_predictions.iloc[:,sample_key] = y_hat_quaids.stack()
        with np.load(r'./output/NN/nfnn/pycoefficients_nfnn'+str(sample_key+1)+'.npz', 'r') as nfnn:
            w_1 = nfnn['w_1']
            w_2 = nfnn['w_2']
            b_1 = nfnn['b_1']
            b_2 = nfnn['b_2']
            y_hat_nfnn = nf.forwardprop(log_inputs, w_1, w_2, b_1, b_2)[0].eval()
            y_hat_nfnn = pd.DataFrame(y_hat_nfnn, columns=y.keys())
            nfnn_predictions.iloc[:,sample_key] = y_hat_nfnn.stack()
        with np.load(r'./output/NN/cfnn/pycoefficients_cfnn'+str(sample_key+1)+'.npz', 'r') as cfnn:
            w_1 = cfnn['w_1']
            w_2 = cfnn['w_2']
            b_1 = cfnn['b_1']
            b_2 = cfnn['b_2']
            y_hat_cfnn = nf.forwardprop(cfnn_inputs, w_1, w_2, b_1, b_2)[0].eval()
            y_hat_cfnn = pd.DataFrame(y_hat_cfnn, columns=y.keys())
            cfnn_predictions.iloc[:,sample_key] = y_hat_cfnn.stack()    
        with np.load(r'./output/NN/cfnn_neg/pycoefficients_cfnn_neg_'+str(sample_key+1)+'.npz', 'r') as cfnn_neg:
            w_1 = cfnn_neg['w_1']
            w_2 = cfnn_neg['w_2']
            b_1 = cfnn_neg['b_1']
            b_2 = cfnn_neg['b_2']
            y_hat_cfnn_neg = nf.forwardprop(cfnn_inputs, w_1, w_2, b_1, b_2)[0].eval()
            y_hat_cfnn_neg = pd.DataFrame(y_hat_cfnn_neg, columns=y.keys())
            cfnn_neg_predictions.iloc[:,sample_key] = y_hat_cfnn_neg.stack()    
    
    aids_results = pd.DataFrame(index = multi_index, columns = ['b95', 'mid', 'u95'])
    aids_results.loc[:,'mid'] = aids_predictions.mean(axis = 1)
    aids_results.loc[:,'b95'] = np.expand_dims(pd.DataFrame(np.sort(aids_predictions)).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
    aids_results.loc[:,'u95'] = np.expand_dims(pd.DataFrame(np.sort(aids_predictions)).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
    
    quaids_results = pd.DataFrame(index = multi_index, columns = ['b95', 'mid', 'u95'])
    quaids_results.loc[:,'mid'] = quaids_predictions.mean(axis = 1)
    quaids_results.loc[:,'b95'] = np.expand_dims(pd.DataFrame(np.sort(quaids_predictions)).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
    quaids_results.loc[:,'u95'] = np.expand_dims(pd.DataFrame(np.sort(quaids_predictions)).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
    
    nfnn_results = pd.DataFrame(index = multi_index, columns = ['b95', 'mid', 'u95'])
    nfnn_results.loc[:,'mid'] = nfnn_predictions.mean(axis = 1)
    nfnn_results.loc[:,'b95'] = np.expand_dims(pd.DataFrame(np.sort(nfnn_predictions)).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
    nfnn_results.loc[:,'u95'] = np.expand_dims(pd.DataFrame(np.sort(nfnn_predictions)).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
    
    cfnn_results = pd.DataFrame(index = multi_index, columns = ['b95', 'mid', 'u95'])
    cfnn_results.loc[:,'mid'] = cfnn_predictions.mean(axis = 1)
    cfnn_results.loc[:,'b95'] = np.expand_dims(pd.DataFrame(np.sort(cfnn_predictions)).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
    cfnn_results.loc[:,'u95'] = np.expand_dims(pd.DataFrame(np.sort(cfnn_predictions)).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
    
    cfnn_neg_results = pd.DataFrame(index = multi_index, columns = ['b95', 'mid', 'u95'])
    cfnn_neg_results.loc[:,'mid'] = cfnn_neg_predictions.mean(axis = 1)
    cfnn_neg_results.loc[:,'b95'] = np.expand_dims(pd.DataFrame(np.sort(cfnn_neg_predictions)).iloc[:,int(0.05*number_bootstrap)-1],axis = 1)
    cfnn_neg_results.loc[:,'u95'] = np.expand_dims(pd.DataFrame(np.sort(cfnn_neg_predictions)).iloc[:,int(0.95*number_bootstrap)-1],axis = 1)
    
    x_axis = np.log(main_data.loc[:,'total.expenditure']) #X-axis of all graphs
    label_list = {'wt.breadcereal': 'Bread and Cereals', 
                  'wt.meat' : 'Meat', 
                  'wt.fish' : 'Fish and Seafood', 
                  'wt.milk' : 'Milk, Cheese, and Eggs',
                  'wt.fat' : 'Oils and Fats', 
                  'wt.fruit' : 'Fruit',
                  'wt.vegetable' : 'Vegetables', 
                  'wt.sugar' : 'Sugar, Jam, Confectionery', 
                  'wt.other' : 'Other', 
                  'wt.nonalc' : 'Non-Alcoholic Beverages'} #Label list for y-axis labeling
    model_list = ['AIDS', 'QUAIDS', 'NFNN', 'CFNN', 'CFNN_NEG']
        
    for w in y.keys():
        
        AIDS_predictions = aids_results.xs(w, level = 'budget')
        QUAIDS_predictions = quaids_results.xs(w, level = 'budget')
        NFNN_predictions = nfnn_results.xs(w, level = 'budget')
        CFNN_predictions = cfnn_results.xs(w, level = 'budget')
        CFNN_NEG_predictions = cfnn_neg_results.xs(w, level = 'budget')
        y_axis = y.loc[:,w]
        y_label = label_list[w]
        
        plot_data = pd.DataFrame( columns = model_list + ['total_expenditure'] )
        plot_data['total_expenditure'] = log_expenditure
        for model in model_list:
            plot_data[model] = eval(model+'_predictions')['mid']
        plot_data['CFNN_NEG Upper Bound'] = CFNN_NEG_predictions['u95']
        plot_data['CFNN_NEG Lower Bound'] = CFNN_NEG_predictions['b95']
        
        sns.set()
        sns.set_style("white")
        sns.set_context("paper")
        _, ax = plt.subplots()
        ax.set(xlim = (log_expenditure.min()*0.95, log_expenditure.max()*1.05 ))
        sns.regplot(x = 'total_expenditure', y = 'AIDS', data = plot_data, 
                    ax = ax, scatter = False, label = 'AIDS', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'QUAIDS', data = plot_data, 
                    ax = ax, scatter = False, label = 'QUAIDS', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'NFNN', data = plot_data, 
                   ax = ax, scatter = False, label = 'ADN', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN', data = plot_data, 
                   ax = ax, scatter = False, label = 'TDN', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN_NEG', data = plot_data, 
                   ax = ax, scatter = False, label = 'TDN-N', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN_NEG Upper Bound', data = plot_data, 
                   ax = ax, scatter = False, label = 'TDN-N 95% CI', lowess= True, color = 'black', line_kws= {'ls':'dashed', 'lw': 1})
        sns.regplot(x = 'total_expenditure', y = 'CFNN_NEG Lower Bound', data = plot_data, 
                   ax = ax, scatter = False, lowess= True, color = 'black', line_kws= {'ls': 'dashed', 'lw': 1})
        sns.despine()
        ax.set(xlabel='Log of Total Food Expenditure', ylabel = y_label)
        plt.legend()
        plt.savefig('./output/figure/negativity/'+w+'_negativity.pdf', close = True)
        sess.close()
        
#%%
if block11_comparison_test_costs_robust:
    sess = tf.InteractiveSession()
    cost_frame = pd.DataFrame(columns = ['ADN', 'TDN', 'ADN-R', 'TDN-R'])
    for sample_key in range(number_bootstrap):
        
        with np.load(r'./output/NN/nfnn/pycoefficients_nfnn'+str(sample_key+1)+'.npz', 'r') as nfnn:
            nfnn = nfnn['test_accuracy']
        with np.load(r'./output/NN/cfnn/pycoefficients_cfnn'+str(sample_key+1)+'.npz', 'r') as cfnn:
            cfnn = cfnn['test_accuracy']
        with np.load(r'./output/NN/nfnn/robust/pycoefficients_nfnnrobust'+str(sample_key+1)+'.npz', 'r') as nfnn_r:
            nfnn_r = nfnn_r['test_accuracy']
        with np.load(r'./output/NN/cfnn/robust/pycoefficients_cfnnrobust'+str(sample_key+1)+'.npz', 'r') as cfnn_r:
            cfnn_r = cfnn_r['test_accuracy']
          
        cost_frame.loc[sample_key, 'ADN'] = float( nfnn )
        cost_frame.loc[sample_key, 'TDN'] = float( cfnn)
        cost_frame.loc[sample_key, 'ADN-R'] = float( nfnn_r )
        cost_frame.loc[sample_key, 'TDN-R'] = float( cfnn_r )
        
    #Set sns for graphs with a common style
    sns.set()
    sns.set_style("white")
    sns.set_context("paper")
    #1. Violin Plot of Cost Values
    fig = sns.violinplot(data = cost_frame, bw = 'silverman', inner = 'box', orient = 'h')
    sns.despine(left = True)
    fig.set_xlabel( r'$C(\Theta, \mathcal{X}_t | \mathcal{X}_e, \mathcal{X}_c)$')
    plt.savefig('./output/figure/violin_robust.pdf', close = True)
        
    sess.close()
    
#%% Large Sample Likelihood Ratio Test
if block12_ls_ll_ratio:
    sess = tf.InteractiveSession()
    test_result_normal = np.zeros(number_bootstrap)
    nfnn_normality = np.zeros(number_bootstrap)
    cfnn_normality = np.zeros(number_bootstrap)
    level_of_significance = 0.05
    for sample_key in range(number_bootstrap):
        #Import Data
        data = pd.HDFStore(r'data/processed/bootstrap_sample'+str(sample_key+1), 'r')
        x_test = np.log( data['x_test'] )
        y_test = data['y_test']

        with np.load(r'./output/NN/nfnn/pycoefficients_nfnn'+str(sample_key+1)+'.npz', 'r') as nfnn:
            w_1_nfnn = nfnn['w_1']
            w_2_nfnn = nfnn['w_2']
            b_1_nfnn = nfnn['b_1']
            b_2_nfnn = nfnn['b_2']
            
        with np.load(r'./output/NN/cfnn/pycoefficients_cfnn'+str(sample_key+1)+'.npz', 'r') as cfnn:
            w_1_cfnn = cfnn['w_1']
            w_2_cfnn = cfnn['w_2']
            b_1_cfnn = cfnn['b_1']
            b_2_cfnn = cfnn['b_2']
        
        yhat_nfnn = pd.DataFrame( (nf.forwardprop(x_test, w_1_nfnn, w_2_nfnn, b_1_nfnn, b_2_nfnn)[0]).eval() ,
                                   index = y_test.index, columns = y_test.columns)
        residual_nfnn = y_test - yhat_nfnn
        residual_nfnn = (residual_nfnn.as_matrix()).ravel()        
        
        yhat_cfnn = pd.DataFrame( (nf.forwardprop(x_test, w_1_cfnn, w_2_cfnn, b_1_cfnn, b_2_cfnn)[0]).eval() ,
                                   index = y_test.index, columns = y_test.columns)
        residual_cfnn = y_test - yhat_cfnn
        residual_cfnn = (residual_cfnn.as_matrix()).ravel()
        
        number_obs = y_test.shape[0]
       
        _,norm_nfnn = sp.stats.normaltest(residual_nfnn)
        _,norm_cfnn = sp.stats.normaltest(residual_cfnn)
        
        if norm_nfnn <=  level_of_significance:
            nfnn_normality[sample_key] = 1
        if norm_cfnn <= level_of_significance:
            cfnn_normality[sample_key] = 1
        
        #With normal distribution
        mu_nfnn, var_nfnn = sp.stats.norm.fit(residual_nfnn)
        mu_cfnn, var_cfnn = sp.stats.norm.fit(residual_cfnn)
       
        log_l_unconstrained_normal = 0
        log_l_constrained_normal = 0
        
        for i in range(number_obs):
            log_l_unconstrained_normal = log_l_unconstrained_normal -0.5*np.log(2*np.pi*var_nfnn) - 0.5*(1/var_nfnn)*( (residual_nfnn[i] - mu_nfnn)**2 )
            log_l_constrained_normal = log_l_constrained_normal -0.5*np.log(2*np.pi*var_cfnn) - 0.5*(1/var_cfnn)*( (residual_cfnn[i] - mu_cfnn)**2 )
        
        ll_ratio_normal = 2*(log_l_unconstrained_normal - log_l_constrained_normal)
        critical_value = sp.stats.chi2.isf(level_of_significance,(number_goods**2-number_goods)/2)
        if ll_ratio_normal < critical_value :
            test_result_normal[sample_key] = 0
        else :
            test_result_normal[sample_key] = 1 #1 indicates rejection of the null that the constraint is not binding
        
        rejected_in_normal = test_result_normal.sum()
        normal_rejected_nfnn = nfnn_normality.sum()
        normal_rejected_cfnn = cfnn_normality.sum()
        
    print("With a normal distribution, the null hypothesis: the constraint is not binding is rejected in %s/%s samples at %s level of significance." \
          %(rejected_in_normal, number_bootstrap, level_of_significance))
    print("NFNN is rejected to be normal in %s samples." %(normal_rejected_nfnn))
    print("CFNN is rejected to be normal in %s samples." %(normal_rejected_cfnn))
    
    sns.kdeplot(residual_cfnn)
    sns.kdeplot(residual_nfnn)
    
#%% 13. Error Terms
if block13_error_terms: #NEEDS FIXING PLOTS AND DATA ARE WRONG"
    import copy
    sess = tf.InteractiveSession()
    
    aids_error = {'Bread and Cereal' : pd.DataFrame( columns = np.arange(1,number_bootstrap+1)),
                  'Meat' : pd.DataFrame( columns = np.arange(1,number_bootstrap+1)),
                  'Fish' : pd.DataFrame( columns = np.arange(1,number_bootstrap+1)),
                  'Milk': pd.DataFrame( columns = np.arange(1,number_bootstrap+1)),
                  'Fat': pd.DataFrame( columns = np.arange(1,number_bootstrap+1)),
                  'Fruit': pd.DataFrame( columns = np.arange(1,number_bootstrap+1)),
                  'Vegetable': pd.DataFrame( columns = np.arange(1,number_bootstrap+1)),
                  'Sugar': pd.DataFrame( columns = np.arange(1,number_bootstrap+1)),
                  'Other': pd.DataFrame( columns = np.arange(1,number_bootstrap+1)),
                  'Non-Alcoholic': pd.DataFrame( columns = np.arange(1,number_bootstrap+1))}
    quaids_error = copy.deepcopy(aids_error)
    adn_error = copy.deepcopy(aids_error)
    tdn_error = copy.deepcopy(aids_error)
    
    goods = ['Bread and Cereal', 'Meat', 'Fish', 'Milk', 'Fat', 'Fruit', 'Vegetable', 'Sugar', 'Other',
                 'Non-Alcoholic']
    
    data = pd.HDFStore(r'data/processed/main_sample', 'r')['main_sample']
    log_price = np.log( data.filter(regex = r'^p.', axis = 1) )
    log_expenditure = np.log( data['total.expenditure'] )
    x = log_price.join(log_expenditure).astype(np.float32)
    y = data.filter(regex = r'^w.', axis = 1)
        
    log_price_r = pandas2ri.py2ri(log_price)
    log_expenditure_r = pandas2ri.py2ri(log_expenditure)
    y_r = pandas2ri.py2ri(y)

    for sample_key in range(number_bootstrap):    
        #Import coefficients
        aids_coef = np.load(r"./output/AIDS/aids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        y_hat_aids = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(aids_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )
        y_hat_aids.index = y.index
        quaids_coef = np.load(r"./output/QUAIDS/quaids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        y_hat_quaids = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(quaids_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )
        y_hat_quaids.index = y.index
        with np.load(r'./output/NN/nfnn/pycoefficients_nfnn'+str(sample_key+1)+'.npz', 'r') as nfnn:
            b_1 = nfnn['b_1']
            b_2 = nfnn['b_2']
            w_1 = nfnn['w_1']
            w_2 = nfnn['w_2']
            adn_predict = pd.DataFrame(nf.forwardprop(x, w_1, w_2, b_1, b_2)[0].eval()).set_index(y.index)
    
        with np.load(r'./output/NN/cfnn/pycoefficients_cfnn'+str(sample_key+1)+'.npz', 'r') as cfnn:
            b_1 = cfnn['b_1']
            b_2 = cfnn['b_2']
            w_1 = cfnn['w_1']
            w_2 = cfnn['w_2']
            tdn_predict = pd.DataFrame(nf.forwardprop(x, w_1, w_2, b_1, b_2)[0].eval()).set_index(y.index)
         
        for i in np.arange(number_goods):
            aids_error[goods[i]].iloc[:,sample_key] = y.iloc[:,i] - y_hat_aids.iloc[:,i]
            quaids_error[goods[i]].iloc[:,sample_key] = y.iloc[:,i] - y_hat_quaids.iloc[:,i]
            adn_error[goods[i]].iloc[:,sample_key] = y.iloc[:,i] - adn_predict.iloc[:,i]
            tdn_error[goods[i]].iloc[:,sample_key] = y.iloc[:,i] - tdn_predict.iloc[:,i]
        
    model_list = ['aids', 'quaids', 'adn', 'tdn']    
    for i in np.arange(number_goods):
        aids_error[goods[i]].values.sort()
        quaids_error[goods[i]].values.sort()
        adn_error[goods[i]].values.sort()
        tdn_error[goods[i]].values.sort()
        
        plot_data = pd.DataFrame( columns = model_list + ['total_expenditure'] )
        plot_data['total_expenditure'] = log_expenditure
        for model in model_list:
            plot_data[model] = eval(model+'_error')[goods[i]].median(axis = 1)
        plot_data['TDN Upper Bound'] = adn_error[goods[i]].loc[:,number_bootstrap*0.95]
        plot_data['TDN Lower Bound'] = adn_error[goods[i]].loc[:,number_bootstrap*0.05]
        
        sns.set()
        sns.set_style("white")
        sns.set_context("paper")
        _, ax = plt.subplots()
        ax.set(xlim = (log_expenditure.min()*0.95, log_expenditure.max()*1.05 ))
        sns.regplot(x = 'total_expenditure', y = 'aids', data = plot_data, 
                    ax = ax, scatter = False, label = 'AIDS', lowess = True)
        sns.regplot(x = 'total_expenditure', y = 'quaids', data = plot_data, 
                    ax = ax, scatter = False, label = 'QUAIDS', lowess = True)
        sns.regplot(x = 'total_expenditure', y = 'adn', data = plot_data, 
                   ax = ax, scatter = False, label = 'ADN', lowess = True)
        sns.regplot(x = 'total_expenditure', y = 'tdn', data = plot_data, 
                   ax = ax, scatter = False, label = 'TDN', lowess = True)
        sns.regplot(x = 'total_expenditure', y = 'TDN Upper Bound', data = plot_data, 
                   ax = ax, scatter = False, label = 'TDN 95% CI', lowess = True,  color = 'black', line_kws= {'ls':'dashed', 'lw': 1})
        sns.regplot(x = 'total_expenditure', y = 'TDN Lower Bound', data = plot_data, 
                   ax = ax, scatter = False, lowess = True, color = 'black', line_kws= {'ls': 'dashed', 'lw': 1})
        plt.plot([-10,10], [0,0], '--', lw = 2)
        sns.despine()
        ax.set(xlabel='Log of Total Food Expenditure', ylabel = 'Error: ' + str(goods[i]) )
        plt.legend()
        plt.savefig('./output/figure/errors/'+str(goods[i])+'.pdf', close = True)
        sess.close()
        
#%% - 14. Income Elasticities
if block14_income_elasticities:
    sess = tf.InteractiveSession()
    #Load Main Data Set
    main_data = pd.HDFStore(r'data/processed/main_sample', 'r')['main_sample']
    number_observation = main_data.shape[0]
    #Adjust data for passing it to R
    price = main_data.filter(regex = r'^p.', axis = 1)
    expenditure = pd.DataFrame(main_data['total.expenditure'])
    demographics = main_data.filter(regex = r'^d.', axis = 1)
    demographics = demographics.drop('d.hh.size', axis = 1)
    price_data = pd.DataFrame(price.median()).T.iloc[np.full(number_observation,0)].reset_index().drop('index', axis = 1)
    elasticity_data = pd.concat([price_data,expenditure], axis = 1)
    
    #Initialize DataFrame
    good = price.keys()
    model = ['AIDS', 'QUAIDS', 'ADN', 'TDN']
    multi_index_row = pd.MultiIndex.from_product([np.arange(number_observation), good, model], names = ['index','good', 'model'])
    multi_index_column = pd.MultiIndex.from_product([np.arange(number_bootstrap)+1, good], names = ['sample', 'good'])
    income_elasticity = pd.DataFrame(index = multi_index_row, columns = np.arange(number_bootstrap)+1 ).sort_index()
    
    #Inputs For AIDS and QUAIDS
    log_price = np.log(elasticity_data.filter(regex = r'^p.', axis = 1))
    log_expenditure = np.log(elasticity_data['total.expenditure'])
    
    for sample_key in np.arange(number_bootstrap):
        aids_coef = np.load(r"./output/AIDS/aids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        quaids_coef = np.load(r"./output/QUAIDS/quaids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
        adn = np.load(r'./output/NN/nfnn/pycoefficients_nfnn'+str(sample_key+1)+'.npz', 'r')
        w_1_adn = adn['w_1']
        w_2_adn = adn['w_2']
        b_1_adn = adn['b_1']
        b_2_adn = adn['b_2']
        tdn = np.load(r'./output/NN/cfnn/pycoefficients_cfnn'+str(sample_key+1)+'.npz', 'r')
        w_1_tdn = tdn['w_1']
        w_2_tdn = tdn['w_2']
        b_1_tdn = tdn['b_1']
        b_2_tdn = tdn['b_2']
        for i in np.arange(number_observation):
            log_price_r = pandas2ri.py2ri(log_price.iloc[[i],:])
            log_expenditure_r = pandas2ri.py2ri(log_expenditure[[i]])
            nfnn_input = np.log(elasticity_data.iloc[[i],:])
            cfnn_input = np.log(nf.impose_homogeneity(elasticity_data.iloc[[i],:]))
            inc_elas, _, _, _ = demand_system_elasticity(aids_coef, log_price_r, log_expenditure_r, model = "AIDS")
            income_elasticity.loc[ (i,slice(None), 'AIDS'), sample_key+1] = inc_elas = pandas2ri.ri2py(inc_elas)
            inc_elas, _, _, _ = demand_system_elasticity(quaids_coef, log_price_r, log_expenditure_r, model = "QUAIDS")
            income_elasticity.loc[ (i, slice(None), 'QUAIDS'), sample_key+1] = pandas2ri.ri2py(inc_elas)
            yhat, z1 = nf.neural_predict(nfnn_input,w_1_adn,w_2_adn,b_1_adn,b_2_adn)
            inc_elas, _, _, _ = nf.slutsky_matrix(nfnn_input, yhat, w_1_adn, w_2_adn, z1, 
                                                                              number_goods, 'NFNN')
            income_elasticity.loc[ (i, slice(None), 'ADN'), sample_key+1] = inc_elas
            yhat, z1 = nf.neural_predict(cfnn_input,w_1_tdn,w_2_tdn,b_1_tdn,b_2_tdn)
            inc_elas, _, _, _ = nf.slutsky_matrix(cfnn_input, yhat, w_1_tdn, w_2_tdn, z1, 
                                                                              number_goods, 'CFNN')
            income_elasticity.loc[ (i, slice(None), 'TDN'), sample_key+1] = inc_elas
        adn.close()
        tdn.close()
            
    income_elasticity.to_pickle("./output/other/income_elasticity_distribution")    
    sess.close()