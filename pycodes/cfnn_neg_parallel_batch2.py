from sklearn import linear_model
from sys import path
from joblib import Parallel, delayed
import tensorflow as tf
import numpy as np
import pandas as pd
import os

try:
    os.chdir("/home/seugurlu/Desktop/SVN/trunk/estimering")
except: 
    os.chdir("/home/serhat/SVN/trunk/estimering") #Server
    
path.append('./scripts/pycodes')
import neural_functions_with_neg as nf

number_goods = 10
tol_coef = 1e-3
tol_iter = 1e-8
tol_stochastic = tol_iter * 100
stochastic_grad_epoch_limit = 50
mini_batch_grad_epoch_limit = 200
penalty_limit = 50
number_cores = 35
sample_range = range(100,200)
batch_size = 200
zeta_multiplier = 1.5

def cfnn(sample_key):
    session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1) 
    sess = tf.InteractiveSession(config = session_conf)
    
    """Input Data"""
    file_handle = "./data/processed/bootstrap_sample"+str( sample_key )
    raw_data = pd.HDFStore(file_handle)
    x_train = raw_data['x_train']; y_train = raw_data['y_train']
    x_cv = raw_data['x_cv']; y_cv = raw_data['y_cv']
    x_test = raw_data['x_test']; y_test = raw_data['y_test']

    """Edit Input Output Data"""
    x_train = np.log( nf.impose_homogeneity(x_train) )#Generate input matrix as homogeneity adjusted log
    x_cv = np.log( nf.impose_homogeneity(x_cv) )#Generate input matrix as homogeneity adjusted log
    x_test = np.log( nf.impose_homogeneity(x_test) )#Generate input matrix as homogeneity adjusted log
    
    x_size = number_goods + 1 #Size of the input layer: + expenditure (1)
    y_size = number_goods #Size of the output layer
    
    '''Initialize Parameters and arrays'''
    hh_start = 2 #Set minimum number of hidden layers
    hh_size = hh_start #Initialize
    window = 5
    hh_end = hh_start + window
    hh_max = 2*number_goods + window
    train_results = pd.DataFrame( columns = ['hh_size', 'train_acc'] )
    cv_results = pd.DataFrame( columns = ['hh_size', 'cv_acc'] )
    test_results = pd.DataFrame( columns = ['hh_size', 'test_acc'] )
    
    '''Set Placeholders & Constants for NN'''
    x = tf.placeholder(tf.float32, shape = [None, x_size])
    y = tf.placeholder(tf.float32, shape = [None, y_size])
    length = tf.placeholder(tf.int32, shape = () )
    
    idx = -1 #Initialize
    while hh_size <= hh_end:
        '''This is the main optimization loop'''
        '''Variable Stopping Criteria for hh_size'''
        idx = idx + 1 
        train_results.loc[idx] = [hh_size,0]
        cv_results.loc[idx] = [hh_size,0]
        test_results.loc[idx] = [hh_size,0]
        zeta = tf.constant(1, dtype = tf.float32) #Symmetry coefficient 1st element
            
        '''Variables'''
        w_1 = tf.Variable(tf.truncated_normal([x_size, hh_size], stddev= 0.1), dtype = tf.float32)
        w_2 = tf.Variable(tf.truncated_normal([hh_size, y_size], stddev= 0.1), dtype = tf.float32)
        b_1 = tf.Variable(tf.random_normal([hh_size]), dtype = tf.float32)
        b_2 = tf.Variable(tf.random_normal([y_size]), dtype = tf.float32)
        
        '''Model'''
        yhat, z1 = nf.forwardprop(x, w_1, w_2, b_1, b_2)
        sym_cost, neg_cost = nf.theoretical_costs(length, x, yhat, w_1, w_2, z1, number_goods, 1, 1)
        train_cost = tf.add( tf.reduce_mean( (y - yhat)**2 ), tf.multiply(zeta,tf.add(sym_cost, neg_cost) ) )
        train_step = tf.train.AdamOptimizer(epsilon=1e-4).minimize(train_cost)
        accuracy_cost = tf.reduce_mean( (y - yhat)**2 ) #To evaluate accuracy
        
        '''Estimation'''
        tf.global_variables_initializer().run() #Initialize variables
        train_accuracy_post_optimization = 10.0 #initialize stopping criterion
        
        #Stochastic Adam
        print("Stochastic gradient starts with hhsize: %s and zeta: %.2f" %(hh_size, zeta.eval()))
        for epoch in range(stochastic_grad_epoch_limit):
            train_accuracy_pre_optimization = train_accuracy_post_optimization
            #Start with stochastic to converge around minimum faster
            for i in range( len(x_train) ):
                train_step.run({ x:x_train[i:i+1], y:y_train[i:i+1], length: 1 })
            train_accuracy_post_optimization = train_cost.eval( {x: x_train, y: y_train, length : len(x_train)} )
            print('Stochastic gradient continues at hhsize: %s, zeta: %.2f, epoch %s. Post_cost: %.8f' %(hh_size, zeta.eval(), epoch, train_accuracy_post_optimization ) )
            if tf.abs( train_accuracy_pre_optimization - train_accuracy_post_optimization ).eval() <= tol_stochastic :
                print("Stochastic gradient converged")
                break
        
        print("Minibatch gradient starts with hhsize: %s and zeta: %.2f" %(hh_size, zeta.eval()))
        for epoch in range(mini_batch_grad_epoch_limit):
            train_accuracy_pre_optimization = train_accuracy_post_optimization
            #Continue with mini batch
            for i in range( int( len(x_train) / batch_size )): # Train with each example
                batch_x = x_train[i*batch_size:(i+1)*batch_size]
                batch_y = y_train[i*batch_size:(i+1)*batch_size]
                train_step.run( {x:batch_x, y:batch_y, length:batch_size} )
            remaining_observations = len(x_train) % batch_size
            if remaining_observations != 0:
                batch_x = x_train[len(x_train)-remaining_observations:len(x_train)]
                batch_y = y_train[len(y_train)-remaining_observations:len(y_train)]
                train_step.run( {x:batch_x, y:batch_y, length:remaining_observations} )
            train_accuracy_post_optimization = train_cost.eval( {x: x_train, y: y_train, length : len(x_train)} )
            print('Minibatch gradient continues at hhsize: %s, zeta: %.2f, epoch %s. Post_cost: %.8f' %(hh_size, zeta.eval(), epoch, train_accuracy_post_optimization ) )
            if tf.abs( train_accuracy_pre_optimization - train_accuracy_post_optimization ).eval() <= tol_iter :
                print("Stochastic gradient converged")
                break
        
        print('starting penalty method')
        for epoch1 in range(penalty_limit):
            zeta = zeta * zeta_multiplier
            pre_coef = tf.constant( tf.concat( [ tf.reshape(w_1, [-1]),
                                                  tf.reshape(w_2, [-1]),
                                                  tf.reshape(b_1, [-1]),
                                                  tf.reshape(b_2, [-1]) ], axis = 0).eval())
                
            print("Minibatch gradient starts with hhsize: %s and zeta: %.2f" %(hh_size, zeta.eval()))
            for epoch2 in range(mini_batch_grad_epoch_limit):
                train_accuracy_pre_optimization = train_accuracy_post_optimization
                for i in range( int( len(x_train) / batch_size )): # Train with each example
                    batch_x = x_train[i*batch_size:(i+1)*batch_size]
                    batch_y = y_train[i*batch_size:(i+1)*batch_size]
                    train_step.run( {x:batch_x, y:batch_y, length:batch_size} )
                remaining_observations = len(x_train) % batch_size
                if remaining_observations != 0:
                    batch_x = x_train[len(x_train)-remaining_observations:len(x_train)]
                    batch_y = y_train[len(y_train)-remaining_observations:len(y_train)]
                    train_step.run( {x:batch_x, y:batch_y, length:remaining_observations} )
                train_accuracy_post_optimization = train_cost.eval( {x: x_train, y: y_train, length : len(x_train)} )
                print('Minibatch gradient continues at hhsize: %s, zeta: %.2f, epoch2 %s. Post_cost: %.8f' %(hh_size, zeta.eval(), epoch2, train_accuracy_post_optimization ) )
                if tf.abs( train_accuracy_pre_optimization - train_accuracy_post_optimization ).eval() <= tol_iter :
                    print("Minibatch gradient converged")
                    break
    
            post_coef = tf.concat([tf.reshape(w_1, [-1]),
                                   tf.reshape(w_2, [-1]),
                                   tf.reshape(b_1, [-1]),
                                   tf.reshape(b_2, [-1])], axis = 0)
                    
            coef_dif = tf.reduce_max( tf.abs( pre_coef - post_coef) )
            print('coef_dif %.6f' %(coef_dif.eval()) )
            
            if coef_dif.eval() <= tol_coef :
                break
            
        train_acc = accuracy_cost.eval( {x: x_train, y: y_train, length:len(x_train)} )
        cv_acc = accuracy_cost.eval( {x: x_cv, y: y_cv, length:len(x_cv)} )
        test_acc = accuracy_cost.eval( {x: x_test, y: y_test, length: len(x_test)} )
        
        #Save Training Optimization
        train_results.loc[idx , 'train_acc'] = train_acc
        cv_results.loc[idx, 'cv_acc'] = cv_acc
        test_results.loc[idx, 'test_acc'] = test_acc

        save_name = "./output/Temp/cfnn_neg/coefficients_cfnn_neg_"+str(sample_key)+"_"+str(hh_size)
        np.savez(save_name, w_1 = w_1.eval(), w_2 = w_2.eval(), b_1 = b_1.eval(), b_2 = b_2.eval(), hh_size = hh_size,
                train_accuracy = train_acc, cv_accuracy = cv_acc, test_accuracy = test_acc )
        #Stop for while
        if hh_size == hh_end:
            reg = linear_model.LinearRegression()
            x_reg = cv_results.loc[:, 'hh_size'].tail(window).values.reshape(window,1)
            y_reg = cv_results.loc[:, 'cv_acc'].tail(window).values.reshape(window,1)
            reg.fit(x_reg,y_reg)
            if reg.coef_ < -tol_iter:
                if hh_end <= hh_max:
                    hh_end += 1
                else:
                    hh_end = hh_end
        
        hh_size += 1
     
    hh_size = int( cv_results.loc[cv_results['cv_acc'].idxmin(), 'hh_size'] )#Size with the best cross-validation result
    with np.load("./output/Temp/cfnn_neg/coefficients_cfnn_neg_"+str(sample_key)+"_"+str(hh_size)+'.npz') as data:
        w_1 = data['w_1']
        w_2 = data['w_2']
        b_1 = data['b_1']
        b_2 = data['b_2']
        hh_size = data['hh_size']
        train_accuracy = data['train_accuracy']
        cv_accuracy = data['cv_accuracy']
        test_accuracy = data['test_accuracy']
    return(train_accuracy, cv_accuracy, test_accuracy, w_1, w_2, b_1, b_2)
    raw_data.close()
    sess.close()
    
#%%
def cfnn_parallel(sample_key):
    train_accuracy, cv_accuracy, test_accuracy, w_1, w_2, b_1, b_2 = cfnn( sample_key )
    np.savez("./output/NN/cfnn_neg/pycoefficients_cfnn_neg_"+str(sample_key),
             train_accuracy = train_accuracy,
             cv_accuracy = cv_accuracy,
             test_accuracy = test_accuracy,
             w_1 = w_1,
             w_2 = w_2,
             b_1 = b_1,
             b_2 = b_2)
    
Parallel(n_jobs = number_cores)(delayed(cfnn_parallel)(sample_key+1) for sample_key in sample_range )
