#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 12:14:58 2017

@author: seugurlu
"""

#    if number_bootstrap == 100:
#        aids1000_coef = np.load(r"./output/AIDS/Sample Sizes/aids"+str(sample_key+1)+"_1000.npy", encoding = 'bytes')[0]
#        y_hat_aids1000 = pd.DataFrame( pandas2ri.ri2py(
#                demand_system_predict(aids1000_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )
#        aids2000_coef = np.load(r"./output/AIDS/Sample Sizes/aids"+str(sample_key+1)+"_2000.npy", encoding = 'bytes')[0]
#        y_hat_aids2000 = pd.DataFrame( pandas2ri.ri2py(
#            demand_system_predict(aids2000_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )
#        aids3000_coef = np.load(r"./output/AIDS/Sample Sizes/aids"+str(sample_key+1)+"_3000.npy", encoding = 'bytes')[0]
#        y_hat_aids3000 = pd.DataFrame( pandas2ri.ri2py(
#            demand_system_predict(aids3000_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )
#        aids4000_coef = np.load(r"./output/AIDS/Sample Sizes/aids"+str(sample_key+1)+"_4000.npy", encoding = 'bytes')[0]
#        y_hat_aids4000 = pd.DataFrame( pandas2ri.ri2py(
#            demand_system_predict(aids4000_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )
#        quaids1000_coef = np.load(r"./output/QUAIDS/Sample Sizes/quaids"+str(sample_key+1)+"_1000.npy", encoding = 'bytes')[0]
#        y_hat_quaids1000 = pd.DataFrame( pandas2ri.ri2py( 
#            demand_system_predict(quaids1000_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )
#        quaids2000_coef = np.load(r"./output/QUAIDS/Sample Sizes/quaids"+str(sample_key+1)+"_2000.npy", encoding = 'bytes')[0]
#        y_hat_quaids2000 = pd.DataFrame( pandas2ri.ri2py( 
#            demand_system_predict(quaids2000_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )
#        quaids3000_coef = np.load(r"./output/QUAIDS/Sample Sizes/quaids"+str(sample_key+1)+"_3000.npy", encoding = 'bytes')[0]
#        y_hat_quaids3000 = pd.DataFrame( pandas2ri.ri2py( 
#            demand_system_predict(quaids3000_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )
#        quaids4000_coef = np.load(r"./output/QUAIDS/Sample Sizes/quaids"+str(sample_key+1)+"_4000.npy", encoding = 'bytes')[0]
#        y_hat_quaids4000 = pd.DataFrame( pandas2ri.ri2py( 
#            demand_system_predict(quaids4000_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )
#        with np.load(r'./output/NN/Size/pycoefficients_nfnn_'+str(sample_key+1)+'_1000.npz', 'r') as nfnn:
#            nfnn1000_test_accuracy = nfnn['test_accuracy']
#    
#        with np.load(r'./output/NN/Size/pycoefficients_nfnn_'+str(sample_key+1)+'_2000.npz', 'r') as nfnn:
#            nfnn2000_test_accuracy = nfnn['test_accuracy']
#    
#        with np.load(r'./output/NN/Size/pycoefficients_nfnn_'+str(sample_key+1)+'_3000.npz', 'r') as nfnn:
#            nfnn3000_test_accuracy = nfnn['test_accuracy']
#        
#        with np.load(r'./output/NN/Size/pycoefficients_nfnn_'+str(sample_key+1)+'_4000.npz', 'r') as nfnn:
#            nfnn4000_test_accuracy = nfnn['test_accuracy']


for sample_key in range(number_bootstrap):
    #Import coefficients
    aids_coef = np.load(r"./output/AIDS/aids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
    prediction_frame.loc[sample_key, 'AIDS'] = pd.DataFrame( pandas2ri.ri2py( 
        demand_system_predict(aids_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )[0][0]
    
    quaids_coef = np.load(r"./output/QUAIDS/quaids"+str(sample_key+1)+".npy", encoding = 'bytes')[0]
    prediction_frame.loc[sample_key, 'QUAIDS'] = pd.DataFrame( pandas2ri.ri2py( 
        demand_system_predict(quaids_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )[0][0]
    
    with np.load(r'./output/NN/pycoefficients_nfnn_'+str(sample_key+1)+'.npz', 'r') as nfnn:
        yhat_nfnn = nf.forwardprop(nfnn_input, nfnn['w_1'], nfnn['w_2'], nfnn['b_1'], nfnn['b_2'])
        prediction_frame.loc[sample_key, 'NFNN'] = sess.run(yhat_nfnn)[0][0][0]
        
    with np.load(r'./output/NN/pycoefficients_'+str(sample_key+1)+'.npz', 'r') as cfnn:
        yhat_cfnn = nf.forwardprop(cfnn_input, cfnn['w_1'], cfnn['w_2'], cfnn['b_1'], cfnn['b_2'])
        prediction_frame.loc[sample_key, 'CFNN'] = sess.run(yhat_cfnn)[0][0][0]
        
    if number_bootstrap == 100:
        aids1000_coef = np.load(r"./output/AIDS/Sample Sizes/aids"+str(sample_key+1)+"_1000.npy", encoding = 'bytes')[0]
        prediction_frame.loc[sample_key, 'AIDS1000']= pd.DataFrame( pandas2ri.ri2py(
                demand_system_predict(aids1000_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )[0][0]
        
        aids2000_coef = np.load(r"./output/AIDS/Sample Sizes/aids"+str(sample_key+1)+"_2000.npy", encoding = 'bytes')[0]
        prediction_frame.loc[sample_key, 'AIDS2000'] = pd.DataFrame( pandas2ri.ri2py(
            demand_system_predict(aids2000_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )[0][0]
        
        aids3000_coef = np.load(r"./output/AIDS/Sample Sizes/aids"+str(sample_key+1)+"_3000.npy", encoding = 'bytes')[0]
        prediction_frame.loc[sample_key, 'AIDS3000'] = pd.DataFrame( pandas2ri.ri2py(
            demand_system_predict(aids3000_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )[0][0]
        
        aids4000_coef = np.load(r"./output/AIDS/Sample Sizes/aids"+str(sample_key+1)+"_4000.npy", encoding = 'bytes')[0]
        prediction_frame.loc[sample_key, 'AIDS4000'] = pd.DataFrame( pandas2ri.ri2py(
            demand_system_predict(aids4000_coef, log_price_r, log_expenditure_r, number_goods, model = "AIDS") ) )[0][0]
        
        quaids1000_coef = np.load(r"./output/QUAIDS/Sample Sizes/quaids"+str(sample_key+1)+"_1000.npy", encoding = 'bytes')[0]
        prediction_frame.loc[sample_key, 'QUAIDS1000'] = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(quaids1000_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )[0][0]
        
        quaids2000_coef = np.load(r"./output/QUAIDS/Sample Sizes/quaids"+str(sample_key+1)+"_2000.npy", encoding = 'bytes')[0]
        prediction_frame.loc[sample_key, 'QUAIDS2000'] = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(quaids2000_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )[0][0]
        
        quaids3000_coef = np.load(r"./output/QUAIDS/Sample Sizes/quaids"+str(sample_key+1)+"_3000.npy", encoding = 'bytes')[0]
        prediction_frame.loc[sample_key, 'QUAIDS3000'] = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(quaids3000_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )[0][0]
        
        quaids4000_coef = np.load(r"./output/QUAIDS/Sample Sizes/quaids"+str(sample_key+1)+"_4000.npy", encoding = 'bytes')[0]
        prediction_frame.loc[sample_key, 'QUAIDS4000'] = pd.DataFrame( pandas2ri.ri2py( 
            demand_system_predict(quaids4000_coef, log_price_r, log_expenditure_r, number_goods, model = "QUAIDS") ) )[0][0]
        with np.load(r'./output/NN/Size/pycoefficients_nfnn_'+str(sample_key+1)+'_1000.npz', 'r') as nfnn:
            yhat_nfnn1000 = nf.forwardprop(nfnn_input, nfnn['w_1'], nfnn['w_2'], nfnn['b_1'], nfnn['b_2'])
            prediction_frame.loc[sample_key, 'NFNN1000'] = sess.run(yhat_nfnn1000)[0][0][0]
    
        with np.load(r'./output/NN/Size/pycoefficients_nfnn_'+str(sample_key+1)+'_2000.npz', 'r') as nfnn:
            yhat_nfnn2000 = nf.forwardprop(nfnn_input, nfnn['w_1'], nfnn['w_2'], nfnn['b_1'], nfnn['b_2'])
            prediction_frame.loc[sample_key, 'NFNN2000'] = sess.run(yhat_nfnn2000)[0][0][0]
    
        with np.load(r'./output/NN/Size/pycoefficients_nfnn_'+str(sample_key+1)+'_3000.npz', 'r') as nfnn:
            yhat_nfnn3000 = nf.forwardprop(nfnn_input, nfnn['w_1'], nfnn['w_2'], nfnn['b_1'], nfnn['b_2'])
            prediction_frame.loc[sample_key, 'NFNN3000'] = sess.run(yhat_nfnn3000)[0][0][0]
        
        with np.load(r'./output/NN/Size/pycoefficients_nfnn_'+str(sample_key+1)+'_4000.npz', 'r') as nfnn:
            yhat_nfnn4000 = nf.forwardprop(nfnn_input, nfnn['w_1'], nfnn['w_2'], nfnn['b_1'], nfnn['b_2'])
            prediction_frame.loc[sample_key, 'NFNN4000'] = sess.run(yhat_nfnn4000)[0][0][0]
            
prediction_frame.filter(['AIDS1000', 'AIDS2000', 'AIDS3000', 'AIDS4000']).plot.kde()
prediction_frame.filter(['NFNN1000', 'NFNN2000', 'NFNN3000', 'NFNN4000']).plot.kde()


for sample_size in [1000, 2000, 3000, 4000]:
        column_name = 'AIDS'+str(sample_size)
        prediction_name = 'y_hat_aids'+str(sample_size)
        cost_frame.loc[sample_key, column_name] = cost(y_test, eval(prediction_name) )
        
        column_name = 'QUAIDS'+str(sample_size)
        prediction_name = 'y_hat_quaids'+str(sample_size)
        cost_frame.loc[sample_key, column_name] = cost(y_test, eval( prediction_name ) )
        
        column_name = 'NFNN'+str(sample_size)
        cost_frame.loc[sample_key, column_name] = float(eval('nfnn'+str(sample_size)+'_test_accuracy') )