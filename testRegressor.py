#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 07:51:11 2017

@author: john
"""
#set working directory
import os
os.chdir('/home/john/rllab_project1/')

import trpoLibFuncs as tFuncs
import matplotlib.pyplot as plt

import lasagne.nonlinearities as NL
import numpy as np
#import cma

#code to test that the regressor process is learning/working correctly

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
#from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor

#from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

def buildGMLP(nonLin):
    regArgs={}
    regArgs['normalize_inputs'] = False
    regArgs['normalize_outputs'] = False
    regArgs['hidden_nonlinearity']=nonLin
    regArgs['hidden_sizes'] = (64,64,8)
    #only used if adaptive_std == True
    regArgs['std_hidden_sizes']= (32,16,16)
    regArgs['adaptive_std'] = False
    regArgs['learn_std']=False

    gMLP_reg = GaussianMLPRegressor(
            input_shape=(1,),
            output_dim=1,
            name="vf1",
           **regArgs
        )
    return gMLP_reg

#build list of "features" -> x values from function
def buildXList(numValues, maxValue):
    #get numValues x values equally spaced between -maxValue and +maxValue
    #and format appropriately for regressor
    #list of lists of single values
    xList=(np.arange(numValues).reshape(-1, 1) / (numValues /( maxValue*2.0))) - maxValue
    return xList

#function to approximate
def yFunc2(x):
    x2= x * x  
    x4 = x2 * x2
    return .1*x4 + (-6 * x * x2) + (2 * x2) + (3 * x) + 1
#function to approximate
def yFunc(x):
    return (1+np.sin(x)) * np.cos(10.0*x)

#build rollout-style paths to consume for faux baselines    
def buildPaths(xList):    
    paths=[]
    
    for x in xList :
        path = dict(
            observations=np.array([x]),   #observations are features here
            rewards=np.array(x),            #only included because linear baseline uses this list as a size argument in formating
            returns=np.array([yFunc(x)])     #returns are functional result here
        )
        paths.append(path)
    return paths
#evaluate function for all vals in testList
def calcActVals(testList):
    yVals=[yFunc(x) for x in testList]
    return yVals

def trainAndFitLinear(testList, paths, linBL):
    #train linear
    linBL.fit(paths)
    yVals=[]
    #use testlist to predict
    for x in testList :
        testPath = dict(
                observations=np.array([x]),   #observations are features here
                rewards=np.array(x),
                )
            
        yVals.append(linBL.predict(testPath).flatten())
    return yVals

def trainAndFitGMLP(testList, paths, numFits, gMLP_Reg):
    #train gaussian
    features = np.concatenate([p["observations"] for p in paths])
    values = np.concatenate([p["returns"] for p in paths])
    gMLP_Reg.resetOptimizer()
    for i in range(numFits):
        gMLP_Reg.fit(features, values.reshape((-1, 1)))
    #use testlist to predict
    yVals=[gMLP_Reg.predict([x]).flatten() for x in testList]
    return yVals


#plot results of various iterations of given gaussian mlp fitting
def buildPltGMLP(xList, paths, gMLP_Reg_relu):
    #input list to test trained regressors
    yVals_raw= [yFunc(x) for x in xList]
    #comment out to ignore linear baseline
#    linBL = LinearFeatureBaseline(env_spec=None)
#    yVals_lin = trainAndFitLinear(xList, paths, linBL)
    
    yVals_gmlp_1 = trainAndFitGMLP(xList, paths, 1, gMLP_Reg_relu)
    yVals_gmlp_10 = trainAndFitGMLP(xList, paths, 9, gMLP_Reg_relu)
    #yVals_gmlp_20 = trainAndFitGMLP(xList, paths, 10, gMLP_Reg_relu)
    yVals_gmlp_50 = trainAndFitGMLP(xList, paths, 40, gMLP_Reg_relu)
    
    #gmlpNet, gmlpNetFunc = tFuncs.buildVFNetFromBaseline(gMLP_Reg_relu)
    #yVals_gmlp_Net = gmlpNetFunc(xList)
    
    plt.plot(xList,yVals_raw,c='r')
#    plt.plot(xList,yVals_lin,c='c')
    plt.plot(xList,yVals_gmlp_1,c='k')
    #plt.plot(xList,yVals_gmlp_Net,c='b')
    plt.plot(xList,yVals_gmlp_10,c='g')
    #plt.plot(xList,yVals_gmlp_20,c='m')
    plt.plot(xList,yVals_gmlp_50,c='y')
    plt.show()

#env.spec not used for Linear bl
#linBL = LinearFeatureBaseline(env_spec=None)
#build gaussian MLP's with different activations
gMLP_Reg_relu = buildGMLP(NL.rectify)
#gMLP_Reg_sig = buildGMLP(NL.sigmoid)

#list of lists of numValues single values, spanning from -maxValue to +maxValue  
xList=buildXList(numValues=1000, maxValue=5.0)
paths=buildPaths(xList)
buildPltGMLP(xList, paths, gMLP_Reg_relu)
#buildPltGMLP(xList, paths, gMLP_Reg_sig)
##input list to test trained regressors
#testList = xList
#yVals_raw= [yFunc(x) for x in testList]
##yVals_lin = trainAndFitLinear(testList, paths, linBL)
#
#yVals_gmlp_relu_1 = trainAndFitGMLP(testList, paths, 1, gMLP_Reg_relu)
#yVals_gmlp_relu_10 = trainAndFitGMLP(testList, paths, 9, gMLP_Reg_relu)
#yVals_gmlp_relu_20 = trainAndFitGMLP(testList, paths, 10, gMLP_Reg_relu)
##yVals_gmlp_sig = trainAndFitGMLP(testList, paths, 1, gMLP_Reg_sig)
#
#plt.plot(testList,yVals_raw,c='r')
##plt.plot(testList,yVals_lin,c='b')
#plt.plot(testList,yVals_gmlp_relu_1,c='g')
#plt.plot(testList,yVals_gmlp_relu_10,c='y')
#plt.plot(testList,yVals_gmlp_relu_20,c='m')
##plt.plot(testList,yVals_gmlp_sig,c='y')
#plt.show()

    
    

##train linear
#linBL.fit(paths)
#
#
##train gaussian
#features = np.concatenate([p["observations"] for p in paths])
#values = np.concatenate([p["returns"] for p in paths])
#gMLP_Reg.fit(features, values.reshape((-1, 1)))
#
###testing values - value we wish to calculate
##testPath = dict(
##        observations=np.array([[.5]]),   #observations are features here
##        rewards=np.array([.5]),
##        )
##
##linBL.predict(testPath).flatten()
##gMLP_Reg.predict(testPath["observations"]).flatten()
#
##list to test regressor predictions
#testList = xList
#
#
#yVals_gmlp = []
#yVals_lin=[]
#for x in testList :
#    testPath = dict(
#            observations=np.array([x]),   #observations are features here
#            rewards=np.array(x),
#            )
#        
#    yVals_lin.append(linBL.predict(testPath).flatten())
#    yVals_gmlp.append(gMLP_Reg.predict(testPath["observations"]).flatten())
#functional evaluation of values in testing list
