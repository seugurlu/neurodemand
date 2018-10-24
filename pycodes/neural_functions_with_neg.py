#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:59:54 2017
Main version
@author: seugurlu
"""
import numpy as np
import tensorflow as tf

#%%BELOW FUNCTIONS ARE ONLY FOR WELFARE CALC WHERE THE INPUT IS A SINGLE OBS
def relu(x):
    if x < 0 :
        output = 0
    else:
        output = x
    return(output)


def softmax(x):
    e_x = np.exp(x - int(np.max(x)))
    return e_x / int(e_x.sum())

def neural_predict(x,w_1,w_2,b_1,b_2): 
    z1 = np.add(np.matmul(x, w_1), b_1)
    a1 = np.vectorize(relu, otypes = [np.float])(z1)  # Activation for the hidden layer function
    z2 = np.add(np.matmul(a1, w_2), b_2)
    yhat = softmax(z2)  # Activation for the output layer
    return yhat, z1
#%% Below here are generic functions
    
def impose_homogeneity(X):
    """Edits Input Matrix to Impose Homogeneity"""
    X2 = X.filter(regex = "^p", axis = 1) **2
    price_length = np.sqrt(X2.sum(axis = 1))
    homogeneous_X = X.div(price_length, axis = 0)
    return homogeneous_X

def relu_grad_calc(x):    
    """Generate RELU gradient values"""
    return( np.greater(x,0).astype( np.float32 ) )

def relu_grad(x):    
    """Generate RELU gradient values"""
    return( tf.cast(tf.greater(x,0), tf.float32 ) )

def theoretical_costs(length, x, yhat, w_1, w_2, z1, number_goods, sym_dev = 1, negativity = 1):
    sym_cost = tf.constant(0, dtype = tf.float32, shape = [1,])
    neg_cost = tf.constant(0, dtype = tf.float32, shape = [1,1])
    obs = tf.constant(0)
    while_condition = lambda obs, sym_cost, neg_cost: tf.less(obs, length)
    def costs(obs, sym_cost, neg_cost):
        x_obs = tf.slice(x,[obs,0],[1,-1])
        yhat_obs = tf.slice(yhat,[obs,0],[1,-1])
        z1_obs = tf.slice(z1,[obs,0],[1,-1])    
        _,_,_,S = slutsky_matrix_tensor(x_obs, yhat_obs, w_1, w_2, z1_obs, number_goods)
        if sym_dev == 1:
            slutsky_dif_matrix = tf.square( S - tf.transpose(S) )
            sym_cost += tf.divide( [tf.reduce_sum( slutsky_dif_matrix )/2], tf.cast(length, dtype = tf.float32) )
        if negativity == 1:
            A = -S
            u = tf.transpose(yhat_obs) - tf.concat([tf.reshape(tf.reduce_sum(yhat_obs**2)**0.5, [1,1]), tf.zeros([number_goods-1,1])],0)
            beta = - tf.matmul(u,u, transpose_a=True) * 0.5
            alpha = beta**(-2) *tf.matmul(u, tf.matmul(A,u), transpose_a= True )
            w = -beta**(-1) * tf.matmul(A,u)
            v = (alpha*0.5)*u-w
            K = A + tf.matmul(u,v,transpose_b=True) + tf.matmul(v,u,transpose_b=True)
            K22 = tf.slice(K,[1,1],[number_goods-1,number_goods-1])
            eig_val = tf.self_adjoint_eig(K22)[0]
            min_eig_val = tf.reduce_min(eig_val)
            neg_cost += tf.divide( tf.maximum(0.,-min_eig_val)**2, tf.cast(length, dtype=tf.float32) )
        return(tf.add(obs,1), sym_cost, neg_cost)
    obs, sym_cost, neg_cost = tf.while_loop(while_condition, costs, [obs, sym_cost, neg_cost])
    return(sym_cost, neg_cost)
    
def forwardprop(x, w_1, w_2, b_1, b_2):
    """Forward Propagation"""
    z1 = tf.add(tf.matmul(x, w_1), b_1)
    a1 = tf.nn.relu(z1)  # Activation for the hidden layer function
    z2 = tf.add(tf.matmul(a1, w_2), b_2)
    yhat = tf.nn.softmax(z2)  # Activation for the output layer
    return yhat, z1

def slutsky_matrix_tensor(x, yhat, w_1, w_2, z1, number_goods):
    diagonal = tf.eye(number_goods, dtype = tf.float32)
    diagonal2 = tf.eye(number_goods+1, dtype = tf.float32)
    matrix_B = tf.tile(-yhat, (number_goods,1))
    matrix_B = tf.add( diagonal, matrix_B)
    matrix_A = tf.transpose( tf.matmul(relu_grad(z1)*w_1[number_goods,:],w_2) )
    mu_i = tf.transpose(yhat)*tf.matmul(matrix_B, matrix_A)
    mu_ij = tf.zeros( (number_goods, 1), dtype = tf.float32)
    for j in range(number_goods):
        p_bar = tf.squeeze( tf.square( tf.exp( tf.slice(x,[0,j],[1,1] ) ) ) )
        homogeneity_adj = tf.reduce_sum( w_1[:number_goods+1]*( diagonal2[j,:] - p_bar )[:,tf.newaxis], axis = 0)
        matrix_A = tf.transpose( tf.matmul(relu_grad(z1)*homogeneity_adj,w_2) )
        mu_ij = tf.concat([mu_ij, tf.transpose(yhat)*tf.matmul(matrix_B, matrix_A)], axis = 1 )
    mu_ij = tf.slice(mu_ij,[0,1],[number_goods, number_goods])
    #Income Elasticity
    income_elas = (mu_i/tf.transpose(yhat)) + 1
    #Uncompensated Price Elasticity
    uncomp_elas = mu_ij/tf.transpose(yhat) - tf.eye(number_goods)
    #Compensated Price Elasticity
    comp_elas = uncomp_elas + tf.matmul(income_elas,yhat)
    #Slutsky Matrix
    slutsky_matrix = comp_elas*tf.transpose(yhat)
    return(income_elas, uncomp_elas, comp_elas, slutsky_matrix)

def slutsky_matrix(x, yhat, w_1, w_2, z1, number_goods, estimation):
    diagonal = np.eye(number_goods, dtype = np.float32)
    diagonal2 = np.eye(number_goods+1, dtype = np.float32)
    matrix_B = np.tile(-yhat, (number_goods,1))
    matrix_B = np.add( diagonal, matrix_B)
    matrix_A = ( np.matmul(relu_grad_calc(z1)*w_1[number_goods,:],w_2) ).T
    mu_i = np.transpose(yhat)*np.matmul(matrix_B, matrix_A)
    mu_ij = np.zeros( (number_goods, 1), dtype = np.float32)
    for j in range(number_goods):
        p_bar = np.square( np.exp( x.iloc[:,j] ) ).squeeze()
        if estimation == 'CFNN':
            homogeneity_adj = np.sum( w_1[:number_goods+1]*( diagonal2[j,:] - p_bar )[:,np.newaxis], axis = 0)
        else:
            homogeneity_adj = w_1[j,:]
        matrix_A = ( np.matmul(relu_grad_calc(z1)*homogeneity_adj,w_2) ).T
        mu_ij = np.concatenate([mu_ij, np.transpose(yhat)*np.matmul(matrix_B, matrix_A)], axis = 1 )
    mu_ij = mu_ij[:,1:]
    #Income Elasticity
    income_elas = (mu_i/yhat.T) + 1
    #Uncompensated Price Elasticity
    uncomp_elas = mu_ij/yhat.T - np.eye(number_goods)
    #Compensated Price Elasticity
    comp_elas = uncomp_elas + np.matmul(income_elas,yhat)
    #Slutsky Matrix
    slutsky_matrix = comp_elas*yhat.T
    return(income_elas, uncomp_elas, comp_elas, slutsky_matrix)
