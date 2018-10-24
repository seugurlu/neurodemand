#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:59:54 2017
Main version
@author: seugurlu
"""
import numpy as np
import tensorflow as tf

#%% Below here are generic functions
    
def impose_homogeneity(X):
    """Edits Input Matrix to Impose Homogeneity"""
    X2 = X.filter(regex = "^p", axis = 1) **2
    price_length = np.sqrt(X2.sum(axis = 1))
    homogeneous_X = X.div(price_length, axis = 0)
    return homogeneous_X

def sigmoid_grad(x):    
    """Generate RELU gradient values"""
    a = tf.nn.sigmoid(x)
    return( a * (1-a) )

def slutsky_dif(length, x, yhat, w_1, w_2, z1, number_goods):
    output = tf.constant(0, dtype = tf.float32)
    obs = tf.constant(0)
    while_condition = lambda obs, output: tf.less(obs, length)
    def body(obs, output):
        x_obs = tf.slice(x,[obs,0],[1,-1])
        yhat_obs = tf.slice(yhat,[obs,0],[1,-1])
        z1_obs = tf.slice(z1,[obs,0],[1,-1])    
    #Calculate mu_i = yhat * matrix_B * matrix_A (For definitions, see the notes)
        diagonal = tf.eye(number_goods, dtype = tf.float32)
        diagonal2 = tf.eye(number_goods+1, dtype = tf.float32)
        matrix_B = tf.tile(-yhat_obs, (number_goods,1))
        matrix_B = tf.add( diagonal, matrix_B)
        matrix_A = tf.transpose( tf.matmul(sigmoid_grad(z1_obs)*w_1[number_goods,:],w_2) )
        mu_i = tf.transpose(yhat_obs)*tf.matmul(matrix_B, matrix_A)
        mu_ij = tf.zeros( (number_goods, 1), dtype = tf.float32)
        for j in range(number_goods):
            p_bar = tf.squeeze( tf.square( tf.exp( tf.slice(x_obs,[0,j],[1,1] ) ) ) )
            homogeneity_adj = tf.reduce_sum( w_1[:number_goods+1]*( diagonal2[j,:] - p_bar )[:,tf.newaxis], axis = 0)
            matrix_A = tf.transpose( tf.matmul(sigmoid_grad(z1_obs)*homogeneity_adj,w_2) )
            mu_ij = tf.concat([mu_ij, tf.transpose(yhat_obs)*tf.matmul(matrix_B, matrix_A)], axis = 1 )
        mu_ij = tf.slice(mu_ij,[0,1],[number_goods, number_goods])
        matrix_C = tf.matmul(mu_i,yhat_obs)
        slutsky_dif_matrix = tf.square( mu_ij - tf.transpose(mu_ij) + matrix_C - tf.transpose(matrix_C) )
        output += tf.divide( tf.reduce_sum( tf.matrix_band_part(slutsky_dif_matrix, 0, -1) ), tf.cast(length, tf.float32) )
        return(tf.add(obs,1), output)
    obs, output = tf.while_loop(while_condition, body, [obs, output])
    return(output)
    
def forwardprop(x, w_1, w_2, b_1, b_2):
    """Forward Propagation"""
    z1 = tf.add(tf.matmul(x, w_1), b_1)
    a1 = tf.nn.sigmoid(z1)  # Activation for the hidden layer function
    z2 = tf.add(tf.matmul(a1, w_2), b_2)
    yhat = tf.nn.softmax(z2)  # Activation for the output layer
    return yhat, z1