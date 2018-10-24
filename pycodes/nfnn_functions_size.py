#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:13:15 2017
CFNN function that runs the estimation and returns estimated coefficients, hh_size, and coefficient values
@author: seugurlu
"""
import tensorflow as tf
import numpy as np
import neural_functions as nf
import os
from sklearn import linear_model
import pandas as pd

def nfnn(h_size, x_size, y_size, x_train, y_train, x_cv, y_cv, x_test, y_test, tol_iter, number_goods):  
    #4.1 PlaceHolders
    x = tf.placeholder(tf.float32, shape = [None, x_size])
    y = tf.placeholder(tf.float32, shape = [None, y_size])
    length = tf.placeholder(tf.int32, shape = () )
    
    w_1 = tf.Variable(tf.truncated_normal([x_size, h_size], stddev= 0.1), dtype = tf.float32)
    w_2 = tf.Variable(tf.truncated_normal([h_size, y_size], stddev= 0.1), dtype = tf.float32)
    b_1 = tf.Variable(tf.random_normal([h_size]), dtype = tf.float32)
    b_2 = tf.Variable(tf.random_normal([y_size]), dtype = tf.float32)
    
    yhat, z1 = nf.forwardprop(x, w_1, w_2, b_1, b_2)
    train_cost = tf.reduce_mean( (y - yhat)**2 )
    train_step = tf.train.AdamOptimizer(1e-4).minimize(train_cost)
    #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(train_cost)
    accuracy_cost = tf.reduce_mean( (y - yhat)**2 ) #To evaluate accuracy
    
    tf.global_variables_initializer().run() #Initialize variables
    train_accuracy = 10
    
    #Stochastic Adam
    for epoch in range(200):
        train_accuracy_pre = train_accuracy
        # Train with each example
        for i in range(len(x_train)):
            train_step.run( {x:x_train[i:i+1], y:y_train[i:i+1], length:1} )
            if i == len(x_train)-1:
                train_accuracy = train_cost.eval( {x: x_train, y: y_train, length:len(x_train)} )
        if tf.abs( train_accuracy_pre - train_accuracy ).eval() <= tol_iter :
            break
        
    #Complete with Batch Adam
    for epoch in range(200):
        train_accuracy_pre = train_accuracy
        # Train with each example
        train_step.run( {x:x_train, y:y_train, length: len(x_train)} )
        train_accuracy = train_cost.eval( {x: x_train, y: y_train, length:len(x_train)} )
        if tf.abs( train_accuracy_pre - train_accuracy ).eval() <= tol_iter :
            cv_acc = accuracy_cost.eval( {x: x_cv, y: y_cv, length:len(x_cv)} )
            test_acc = accuracy_cost.eval( {x: x_test, y: y_test, length: len(x_test)} )
            break
        elif epoch == 199:
            cv_acc = accuracy_cost.eval( {x: x_cv, y: y_cv, length:len(x_cv)} )
            test_acc = accuracy_cost.eval( {x: x_test, y: y_test, length: len(x_test)} )

    return(cv_acc, test_acc, w_1.eval(), w_2.eval(), b_1.eval(), b_2.eval())

def main_with_cv(sample_key, x_train, y_train, x_cv, y_cv, x_test, y_test, tol_iter, number_goods):
    session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)     
    sess = tf.InteractiveSession(config = session_conf)
    seed = 1
    tf.set_random_seed(seed)
    x_size = len(x_train.columns)
    y_size = number_goods
    
    #Estimation with cross_validation
    hh_size = 1
    hh_end = 10
    window = 10
    cv_results = pd.DataFrame(np.zeros([1, 2]), columns = ['hh_size', 'cv_cost'])
    while hh_size <= hh_end:
        cv_results.loc[hh_size-1] = [hh_size,0]
        cv_accuracy, test_accuracy, w_1_out, w_2_out, b_1_out, b_2_out = nfnn(hh_size, x_size, y_size, x_train, y_train, x_cv, y_cv, x_test, y_test, tol_iter, number_goods)
        save_name = os.getcwd()+"/output/Temp/Size/coefficients_nfnn_size_"+str(sample_key)+"_"+str(hh_size)
        np.savez(save_name, w_1 = w_1_out, w_2 = w_2_out, b_1 = b_1_out, b_2 = b_2_out, hh_size = hh_size,
                 cv_accuracy = cv_accuracy, test_accuracy = test_accuracy )
        cv_results.iloc[hh_size-1,1] = cv_accuracy
        if hh_size == hh_end:
            reg = linear_model.LinearRegression()
            x = cv_results.loc[hh_size-window:hh_size, 'hh_size'].values.reshape(window,1)
            y = cv_results.loc[hh_size-window:hh_size, 'cv_cost'].values.reshape(window,1)
            reg.fit(x,y)
            if reg.coef_ < 0:
                hh_end += 1
        hh_size += 1
            
    hh_size = cv_results['cv_cost'].idxmin() + 1 #Size with the best cross-validation result
    with np.load(os.getcwd()+"/output/Temp/Size/coefficients_nfnn_size_"+str(sample_key)+"_"+str(hh_size)+'.npz') as data:
        w_1_out = data['w_1']
        w_2_out = data['w_2']
        b_1_out = data['b_1']
        b_2_out = data['b_2']
        hh_size = data['hh_size']
        cv_accuracy = data['cv_accuracy']
        test_accuracy = data['test_accuracy']
                    
    return(cv_accuracy, test_accuracy, w_1_out, w_2_out, b_1_out, b_2_out, hh_size)
    
    sess.close()
