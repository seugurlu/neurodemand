#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:02:20 2017

@author: seugurlu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 22:21:55 2017
This file is the output script for the paper.
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
#number_bootstrap = 200
number_bootstrap = 100
number_cores = 35

#Set R functions
demand_system = importr('demand.tools')
demand_system_predict = demand_system.demand_system_predict
demand_system_elasticity = demand_system.demand_elasticity

#Define functions, set arrays
def cost(y_test, y_hat):
    return( 
            np.square( np.subtract(y_test, y_hat) ).stack().mean() 
            )

cost_frame = pd.DataFrame(columns = ['AIDS', 'QUAIDS', 'NFNN', 'CFNN', 'CFNN_NEG'])

#Set Analysis
block1_comparison_test_costs = 0
block2_plotting_demand_functions = 0
block3_estimating_welfare_impact = 0
block4_plotting_welfare_impact = 1
block5_comparison_test_costs_demographics = 0
block6_plotting_demand_functions_demographics = 0
block7_elasticities = 0
block8_table_elasticities = 0
#%%
'''Result 1 - Standard Estimation - Comparison of Test Costs'''
if block1_comparison_test_costs:
    sess = tf.InteractiveSession()
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
        cost_frame.loc[sample_key, 'NFNN'] = float( nfnn )
        cost_frame.loc[sample_key, 'CFNN'] = float( cfnn)
        cost_frame.loc[sample_key, 'CFNN_NEG'] = float( cfnn_neg)
        data.close()
        
    #Set sns for graphs with a common style
    sns.set()
    sns.set_style("white")
    sns.set_context("paper")
    #1. Violin Plot of Cost Values
    fig = sns.violinplot(data = cost_frame, bw = 'silverman', inner = 'point', orient = 'h')
    sns.despine(left = True)
    fig.set_xlabel( r'$C(\Theta, \mathcal{X}_t | \mathcal{X}_e, \mathcal{X}_c)$')
    plt.savefig('./output/figure/violin_standard.pdf', close = True)
        
    #Kolmogorov-Smirnov Tests
    print('NFNN-CFNN KS:', sp.stats.ks_2samp(cost_frame['NFNN'], cost_frame['CFNN']))
    print('AIDS-QUAIDS KS:', sp.stats.ks_2samp(cost_frame['AIDS'], cost_frame['QUAIDS']))
    print('AIDS-NFNN KS:', sp.stats.ks_2samp(cost_frame['AIDS'], cost_frame['NFNN']))
    print('AIDS-CFNN KS:', sp.stats.ks_2samp(cost_frame['AIDS'], cost_frame['CFNN']))
    print('QUAIDS-NFNN KS:', sp.stats.ks_2samp(cost_frame['QUAIDS'], cost_frame['NFNN']))
    print('QUAIDS-CFNN KS:', sp.stats.ks_2samp(cost_frame['QUAIDS'], cost_frame['CFNN']))
    print('QUAIDS-CFNN_NEG KS:', sp.stats.ks_2samp(cost_frame['QUAIDS'], cost_frame['CFNN_NEG']))
    print('CFNN-CFNN_NEG KS:', sp.stats.ks_2samp(cost_frame['CFNN'], cost_frame['CFNN_NEG']))
    
    sess.close()
#%%
if block2_plotting_demand_functions:
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
                   ax = ax, scatter = False, label = 'NFNN', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN', data = plot_data, 
                   ax = ax, scatter = False, label = 'CFNN', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN Upper Bound', data = plot_data, 
                   ax = ax, scatter = False, label = 'CFNN 95% CI', lowess= True, color = 'black', line_kws= {'ls':'dashed', 'lw': 1})
        sns.regplot(x = 'total_expenditure', y = 'CFNN Lower Bound', data = plot_data, 
                   ax = ax, scatter = False, lowess= True, color = 'black', line_kws= {'ls': 'dashed', 'lw': 1})
        sns.despine()
        ax.set(xlabel='Log of Total Expenditure', ylabel = y_label)
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
                   ax = ax, scatter = False, label = 'CFNN', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN Upper Bound', data = plot_data, 
                   ax = ax, scatter = False, label = 'CFNN 95% CI', lowess= True, color = 'black', line_kws= {'ls':'dashed', 'lw': 1})
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
    cost_frame_dem = pd.DataFrame(columns = ['AIDS', 'QUAIDS', 'NFNN', 'CFNN'])
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
        cost_frame_dem.loc[sample_key, 'NFNN'] = float( nfnn )
        cost_frame_dem.loc[sample_key, 'CFNN'] = float( cfnn)
        data.close()
        
    #Set sns for graphs with a common style
    sns.set()
    sns.set_style("white")
    sns.set_context("paper")
    #1. Violin Plot of Cost Values
    fig = sns.violinplot(data = cost_frame_dem, bw = 'silverman', inner = 'point', orient = 'h')
    sns.despine(left = True)
    fig.set_xlabel( r'$C(\Theta, \mathcal{X}_t | \mathcal{X}_e, \mathcal{X}_c)$')
    plt.savefig('./output/figure/violin_standard_demographics.pdf', close = True)
        
    #Kolmogorov-Smirnov Tests
    print('NFNN-CFNN KS:', sp.stats.ks_2samp(cost_frame_dem['NFNN'], cost_frame_dem['CFNN']))
    print('AIDS-QUAIDS KS:', sp.stats.ks_2samp(cost_frame_dem['AIDS'], cost_frame_dem['QUAIDS']))
    print('AIDS-NFNN KS:', sp.stats.ks_2samp(cost_frame_dem['AIDS'], cost_frame_dem['NFNN']))
    print('AIDS-CFNN KS:', sp.stats.ks_2samp(cost_frame_dem['AIDS'], cost_frame_dem['CFNN']))
    print('QUAIDS-NFNN KS:', sp.stats.ks_2samp(cost_frame_dem['QUAIDS'], cost_frame_dem['NFNN']))
    print('QUAIDS-CFNN KS:', sp.stats.ks_2samp(cost_frame_dem['QUAIDS'], cost_frame_dem['CFNN']))
    
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
                   ax = ax, scatter = False, label = 'NFNN', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN', data = plot_data, 
                   ax = ax, scatter = False, label = 'CFNN', lowess= True)
        sns.regplot(x = 'total_expenditure', y = 'CFNN Upper Bound', data = plot_data, 
                   ax = ax, scatter = False, label = 'CFNN 95% CI', lowess= True, color = 'black', line_kws= {'ls':'dashed', 'lw': 1})
        sns.regplot(x = 'total_expenditure', y = 'CFNN Lower Bound', data = plot_data, 
                   ax = ax, scatter = False, lowess= True, color = 'black', line_kws= {'ls': 'dashed', 'lw': 1})
        sns.despine()
        ax.set(xlabel='Log of Total Expenditure', ylabel = y_label)
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
        ax.set(xlabel='Log of Total Expenditure', ylabel = y_label)
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
        ax.set(xlabel='Log of Total Expenditure', ylabel = y_label)
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
    income_elas_frame = pd.DataFrame(index = multi_index, columns = ['AIDS', 'QUAIDS', 'NFNN', 'CFNN'])
    for good in income_elasticity.index.levels[0]:
        income_elas_frame.xs(good, level = 'Good').loc[:,'AIDS'] = aids_estimates.loc[good,:]
        income_elas_frame.xs(good, level = 'Good').loc[:,'QUAIDS'] = quaids_estimates.loc[good,:]
        income_elas_frame.xs(good, level = 'Good').loc[:,'NFNN'] = nfnn_estimates.loc[good,:]
        income_elas_frame.xs(good, level = 'Good').loc[:,'CFNN'] = cfnn_estimates.loc[good,:]
    
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
    for good in income_elasticity.index.levels[0]:
        fig = sns.violinplot ( data = income_elas_frame.xs(good, level = 'Good'), bw = 'silverman', orient = 'h')
        sns.despine(left = True)
        fig.set_xlabel( r'Income Elasticity of '+ label_list[good])
        plt.savefig('./output/figure/violin_income_elas_'+good+'.pdf', close = True)
        plt.close()
    
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