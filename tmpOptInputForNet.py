#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:09:28 2017

@author: john
"""
##stack overflow question on optimizing input for trained neural net
##https://stackoverflow.com/questions/41202399/theano-hessian-assertionerror-tensor-hessian-t-grad-works-but-not-theano-gradi
#import os
#os.environ["THEANO_FLAGS"] = "optimizer=None"
#import theano
#import theano.tensor as T
#import theano.tensor.nnet as nnet
#from theano.tensor.nlinalg import matrix_inverse, det
#import numpy as np
#
#
#def grad_desc(cost, theta):
#    alpha = 0.1 #learning rate
#    return theta - (alpha * T.grad(cost, wrt=theta))
#
#
#in_units = 2
#hid_units = 3
#out_units = 1
#
#b = np.array([[1]], dtype=theano.config.floatX)
#rand_init = np.random.rand(in_units, 1)
#rand_init[0] = 1
#x_sh = theano.shared(np.array(rand_init, dtype=theano.config.floatX))
#th1 = T.dmatrix()
#th2 = T.dmatrix()
#
#nn_hid = T.nnet.sigmoid( T.dot(th1, T.concatenate([x_sh, b])) )
#nn_predict = T.sum( T.nnet.sigmoid( T.dot(th2, T.concatenate([nn_hid, b]))))
#
#
#fc2 = (nn_predict - 1)**2 
#
##This works well with grad_desc
#cost2 = theano.function(inputs=[th1, th2], outputs= fc2, updates=[
#        (x_sh, grad_desc(nn_predict, x_sh))])
#run_forward = theano.function(inputs=[th1, th2], outputs=nn_predict)
#
#cur_cost = 0
#for i in range(10000):
#
#    cur_cost = cost2(theta1.get_value().T, theta2.get_value().T) #call our Theano-compiled cost function, it will auto update weights
#    if i % 500 == 0: #only print the cost every 500 epochs/iterations (to save space)
#        print('Cost: %s | %s' % (cur_cost,x_sh.get_value()))


##############################################################################################
#           original example with error/missing code
##############################################################################################
 #https://stackoverflow.com/questions/41091113/calculate-optimal-input-of-a-neural-network-with-theano-by-using-gradient-desce
     
        

#1. train network to obtain weights
#2. define network function with fixed weights as params, f(x) => nn function on x
#3. apply gradient descent on nn function f(x) to obtain x that maxes f(x) wrt to weights
import os
os.environ["THEANO_FLAGS"] = "optimizer=None"
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np


#step 1 train net get weights
x = T.dvector()
y = T.dscalar()

def layer(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x) #theta1: 3x3 * x: 3x1 = 3x1 ;;; theta2: 1x4 * 4x1
    h = nnet.sigmoid(m)
    return h

def grad_desc(cost, theta):
    alpha = 0.1 #learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))

in_units = 2
hid_units = 3
out_units = 1

theta1 = theano.shared(np.array(np.random.rand(in_units + 1, hid_units), dtype=theano.config.floatX)) # randomly initialize
theta2 = theano.shared(np.array(np.random.rand(hid_units + 1, out_units), dtype=theano.config.floatX))

hid1 = layer(x, theta1) #hidden layer

out1 = T.sum(layer(hid1, theta2)) #output layer
fc = (out1 - y)**2 #cost expression

cost = theano.function(inputs=[x, y], outputs=fc, updates=[
        (theta1, grad_desc(fc, theta1)),
        (theta2, grad_desc(fc, theta2))])
run_forward = theano.function(inputs=[x], outputs=out1)

inputs = np.array([[0,1],[1,0],[1,1],[0,0]]).reshape(4,2) #training data X
exp_y = np.array([1, 0, 0, 0]) #training data Y
cur_cost = 0
for i in range(5000):
    for k in range(len(inputs)):
        cur_cost = cost(inputs[k], exp_y[k]) #call our Theano-compiled cost function, it will auto update weights

print(run_forward([0,1]))

#step 2 use weights to build function
b1 = np.array([[1]], dtype=theano.config.floatX)
b2 = np.array([[1]], dtype=theano.config.floatX)
#b_sh = theano.shared(np.array([[1]], dtype=theano.config.floatX))
rand_init = np.random.rand(in_units, 1)
rand_init[0] = 1
x_sh = theano.shared(np.array(rand_init, dtype=theano.config.floatX))
th1 = T.dmatrix()
th2 = T.dmatrix()

nn_hid = T.nnet.sigmoid( T.dot(th1, T.concatenate([x_sh, b1])) )
nn_predict = T.sum( T.nnet.sigmoid( T.dot(th2, T.concatenate([nn_hid, b2]))))

#step 3 :apply gradient descent on nn function f(x) to obtain x that maxes f(x) wrt to weights
fc2 = (nn_predict - 1)**2

cost3 = theano.function(inputs=[th1, th2], outputs=fc2, updates=[
        (x_sh, grad_desc(fc2, x_sh))])
run_forward = theano.function(inputs=[th1, th2], outputs=nn_predict)

cur_cost = 0
for i in range(10000):

    cur_cost = cost3(theta1.get_value().T, theta2.get_value().T) #call our Theano-compiled cost function, it will auto update weights
    if i % 500 == 0: #only print the cost every 500 epochs/iterations (to save space)
        print('Cost: %s | %s' % (cur_cost,x_sh.get_value()))