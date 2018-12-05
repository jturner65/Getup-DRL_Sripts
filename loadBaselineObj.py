#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:47:15 2017

@author: john
"""
#set working directory
import os
os.chdir('/home/john/rllab_project1/')

import trpoLibFuncs as tFuncs

import lasagne
import lasagne.layers as L
import theano
import theano.tensor as T


##suggested by visak, method for simplifying the representation of the baseline NN
expDict = tFuncs.buildExpDict()
env, polDict, trainDict = tFuncs.buildExperiment(expDict)
baseline=polDict['baseline']

blLayerShapes = baseline._regressor.get_param_shapes()

blParams = L.get_all_param_values(baseline._regressor._mean_network.output_layer)

    
from rllab.core.network import MLP
net = MLP(input_shape=(blLayerShapes[0][0],),
            output_dim=1,
            hidden_sizes=expDict['mlpArch'],
            hidden_nonlinearity=lasagne.nonlinearities.rectify,
            output_nonlinearity=None,
            )
    
L.set_all_param_values(net.output_layer,blParams)
X = net.input_layer.input_var

pred = L.get_output(net.output_layer,deterministic=True)
valueFunc = theano.function([X],pred)



#Third : You can then just query the value of a state using this

vf = valueFunc(observations)