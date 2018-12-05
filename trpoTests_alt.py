#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 11:44:03 2017

@author: john
"""
#set working directory
import os
os.chdir('/home/john/rllab_project1/')

import trpoLibFuncs as tFuncs
import numpy as np
import matplotlib.pyplot as plt

dataDirName=None  #default
#dataDirName = 'experiment_2017_11_30_19_39_12_0001'
#dataDirName = 'experiment_2017_12_01_00_39_57_0001'
#dataDirName = 'experiment_2017_12_01_03_55_12_0001'
################################33
#using iterative BL fitting
#dataDirName = 'experiment_2017_12_04_16_09_23_0001'
#using iterative BL fitting on bl with arch  : 16-16-8
#dataDirName = 'experiment_2017_12_04_21_13_05_0001'     

#name of csv file holding pretrained states
initStatesCSVName='testStateFile.csv' 
#initStatesCSVName='testStateFile2.csv' 

#dataDirName = 'experiment_2017_12_17_05_37_47_0001'
#dataDirName = 'experiment_2017_12_17_06_14_39_0001'
#dataDirName = 'experiment_2017_12_17_08_06_21_0001'
#dataDirName = 'experiment_2017_12_17_08_06_21_0002'
#dataDirName = 'experiment_2017_12_17_08_06_21_0003'
#below was demonstrated to work using opt force suggestion - most demoed as of 2/1/18
#dataDirName = 'experiment_2018_01_19_21_48_47_0001'
#below is 250 iters training
#dataDirName = 'experiment_2018_01_21_20_01_42_0001'
#retrained with slightly different raise_vel 
dataDirName = 'experiment_2018_02_05_06_48_23_0001'
#dataDirName = 'experiment_2018_02_05_08_48_04_0001'

#set all experiment variables (like policy directory) in buildExpDict
#envName='DartStandUp2dTorque-v1'
expDict = tFuncs.buildExpDict(dataDirName=dataDirName,initStatesCSVName=initStatesCSVName)
#print('Directory : {}'.format(expDict['ovrd_dataDirName']))
#need to build env with snapshots enabled to make video
env, polDict, trainDict = tFuncs.buildExperiment(expDict)
#print (expDict['subDirList'])

#uncomment to start training in single-thread
#algo = tFuncs.trainPolicy(env, polDict, trainDict)


#uncomment to build and save a file of initial states to use
#_ = tFuncs.saveNumRndStatesForSkel(env, expDict['savedInitStates'],numStates=50)


#observation != just state - 

#read in weights of trained baseline, build duplicate network based on mean network
#net is value function network, valueFunc is theano function to call with state to get value
#pass regressor from trained baseline
def testBLPredsInPolicy(env, trainDict, polDict, expDict):
    #for ordered dict
    import collections
    
    #load predetermined q and qdot as init states
    #idxsToUse is a list of indexes in precalced state/statedot to use - set to none to use all
    #idxsToUse = None
    idxsToUse = [0]
    #poorly performing initial states
    #idxsToUse = [13,22,28,30,43,44]

    initQList, initQdotList = tFuncs.loadInitStates(env,expDict,idxsToUse=idxsToUse)    

    #get theano-derived net values and jacobians from baseline
    vfRD = tFuncs.buildVFNetFromBaseline(polDict['baseline']._regressor)
    # number of random forces to explore for each initial state to find optimal result in value function
    numRndFrc=50
    #results for all initial states being used - list of each state's resDict sorted dictionary of optimal forces and VF-predicted scores
    vfResAllStates = getOptFrcForAllStates(env,vfRD, idxsToUse, initQList,initQdotList, numRndFrc)
        #for precalced state 0 :
        #run 1 opt val/force : 628.0624722323355 : (45.550352416498392, 63.629717866728214)
        #run 2 opt val/force : 628.0610050278963 : (45.547311670047947, 63.618796886504178)
    #by here we have resAllStates : list of result dictionaries of sorted scores/forces from all init states' 
    #access best value for each initial state by : list(resAllStates[st].items())[0] -> gives tuple of score/force for highest score    
    
    #now use force predictions from baselines to rollout each initial state
    #list of dicts of return values for forces from VF predictions for all states
    roRetsForVFPreds = []
    print('')
    #for each initial state
    for st in range(len(initQList)):
        #set initial q,qdot, frc, and perform rollout
        q = initQList[st]
        qdot = initQdotList[st]
        vfResDict = vfResAllStates[st]
        #rollout results - keyed by value function score, value is rollout score
        resRO = dict()
        #for each force value in descending order of predicted score
        for k,v in vfResDict.items():
            #k is predicted score, v is tuple of x/y force (NOT multiplier)
            #set force multiplier for 
            #env.wrapped_env.env.unwrapped.setFrcMultFromFrc(v[0],v[1])
            tFuncs.setFrcVals(env, v[0],v[1], False)
            env.wrapped_env.env.unwrapped.setNewState(q,qdot)    
            #perform rollout
            path, path_baselines = tFuncs.usePolicy(env, trainDict, polDict, detPolicy=True, renderRes=False, debug=False)
            #get return for this initial state and force value
            resRO[k]=path['returns'][0]
            obs = env.wrapped_env.env.unwrapped._get_obs()
            print('Rollout : Obs Force : {:5.5f},{:5.5f} | fed force :  {:5.5f},{:5.5f}| return: {} '.format(obs[-2],obs[-1],v[0],v[1],resRO[k]))
        #here we have scores of all vf predictions provided for particular state - order them by prediction values
        #resRO_ordered = collections.OrderedDict(sorted(resRO.items(), reverse=True))
        roRetsForVFPreds.append(resRO)      
        
    print('')
    #by here we have all rollout results for all predicted force values for all initial states
    #display results
    for st in range(len(initQList)):
        resRODict = roRetsForVFPreds[st]
        resRO_ordered = collections.OrderedDict(sorted(resRODict.items(), reverse=True))
        vfResDict = vfResAllStates[st]
        for k,v in resRO_ordered.items():
            s = vfResDict[k]
            print('Force : {:5.5f},{:5.5f} |  VF pred return = {:5.5f} | Rollout Act Return = {:5.5f}'.format(s[0],s[1], k, v))
        

#return a list of optimal force dictionaries for all states passed
def getOptFrcForAllStates(env,vfRD, idxsToUse, initQList,initQdotList, numRndFrc, debug=False):
    #results for all initial states being used - list of each state's resDict sorted dictionary of optimal forces and VF-predicted scores
    vfResAllStates = list()
    #for 50 iterations - find optimal result from 50 different starting forces, keeping q and qdot static
    for st in range(len(initQList)):
        if(idxsToUse is not None):
            stIdx = idxsToUse[st]
        else :
            stIdx = st
        q = initQList[st]
        qdot = initQdotList[st]
        #pass q/qdot to find ordered dictionary of best force proposals from value function
        resDict = tFuncs.findVFOptFrcForSt(env,vfRD, q, qdot, numRndFrc)        
        if(debug):
            for k,v in resDict.items():
                print('For state IDX {} :: best VF Pred score : {} | force : {}'.format(stIdx,k,v))
        #optimal force according to value function : 
        #score,optForce = list(resDict.items())[0]
        vfResAllStates.append(resDict) 
        
    return vfResAllStates
        
#retrain baseline with select examples to attempt to improve performance
def retrainBLWithExamples(env, trainDict, polDict, expDict):
    #load a copy of policy and baseline, so that copy baseline can be trained
    newPolDict = tFuncs.loadNewPolDict(expDict)
    #dictionary of retraining arguments
    rtrnDict ={}
    # number of policy rollouts to use to retrain
    rtrnDict['runIters'] = 1000
    #threshold path length for a rollout to be considered "bad"
    rtrnDict['badPathLen'] = 2
    # % of good rollouts to keep (increases # of bad rollouts to retrain bl with)
    rtrnDict['keepRate'] = 0.2
    #set to true to reject all rollouts with path lengths > rejBadPathLen
    rtrnDict['rejectBad'] = False
    #iterations to retrain policy (# of runs of "fit")
    rtrnDict['retrainIters'] = 5
    #whether or not using deterministic policy
    rtrnDict['detPolicy'] = expDict['testWithDetPol']
    #draw results
    rtrnDict['renderRes'] = False
    rtrnDict['setFrc'] = True
    #bounds of force to set - TODO change to list of values?
    rtrnDict['forceToSet'] = ()
    
    #setFrc restricts force used per iter to be in region of CMA-predicted best choice
    newBaseLine, paths, path_bls = tFuncs.retrainBaseline(env, trainDict, newPolDict, rtrnDict)
    newPolDict['baseline']=newBaseLine
    #updated baseline
    tFuncs.testMaxBLAgainstRet(env, trainDict, newPolDict, expDict)
    #orig base line
    tFuncs.testMaxBLAgainstRet(env, trainDict, polDict, expDict)    




#direct access testing
def testPolicy(env, trainDict, polDict, expDict):
    #initial state and state dot 
    #idxsToUse is a list of indexes in precalced state/statedot to use - set to none to use all
    idxsToUse = None
    #idxsToUse = [0]
    initQList, initQdotList = tFuncs.loadInitStates(env,expDict,idxsToUse=idxsToUse)
    for i in range(len(initQList)):
        env.wrapped_env.env.unwrapped.setNewState(initQList[i],initQdotList[i])  
        #env.wrapped_env.env.unwrapped.setDebugMode(True)
        path,_ = tFuncs.usePolicy(env, trainDict, polDict, expDict['testWithDetPol'], renderRes=True)
        #env.wrapped_env.env.unwrapped.frcOffset
        tmpPath = dict(
            observations=np.array(path['observations']),
            #included for linear baselines, which uses length of rewards list to format coefficients vector size
            rewards = np.array(path['rewards'])
        )
        path_baseline = polDict['baseline'].predict(tmpPath)
        
        xAxis = np.linspace(0,len(path_baseline),len(path_baseline),endpoint=False)
        plt.plot(xAxis,path_baseline,c='b', label='Value Function')
        plt.plot(xAxis,path['returns'],c='r', label='Return')
    #    plt.legend(
    #           loc='lower left',
    #           ncol=4,
    #           fontsize=8)
        plt.title('Comparison of Actual Returns and Value Function Predictions\nfor {}'.format(tFuncs.getExpNameDisp(expDict)))
        plt.xlabel('Rollout Step')
        plt.ylabel('Return')
        plt.show()
    
    
        
#show results of value function predictions for all initial states
def showVFPerfForAllStates(env, trainDict, polDict, expDict):
    #initial state and state dot 
    #idxsToUse is a list of indexes in precalced state/statedot to use - set to none to use all
    idxsToUse = None
    #idxsToUse = [0]
    #poorly performing initial states
    #idxsToUse = [13,22,28,30,43,44]
    initQList, initQdotList = tFuncs.loadInitStates(env,expDict,idxsToUse=idxsToUse)
    
    vfRD = tFuncs.buildVFNetFromBaseline(polDict['baseline']._regressor)
    # number of random forces to explore for each state to find optimal result in value function
    numRndFrc=2

    #for consistency - get same force for each state to use as initial guess
    optIFrcForAllSt = getOptFrcForAllStates(env,vfRD, idxsToUse, initQList,initQdotList, numRndFrc)

    returnsPerState = []
    returnsVFPerState = []
    returnsVFPerBad = []
    #test with choosing only initial force
    #env.wrapped_env.env.unwrapped.setDebugMode(True)
    for i in range(len(initQList)):
        #set initial state/state_dot
        env.wrapped_env.env.unwrapped.setNewState(initQList[i],initQdotList[i]) 
        #set optimal initial force given state
        #tFuncs.setOptInitForce(env, initQList[i],initQdotList[i], vfRD, numRndFrc=numRndFrc)    
        _,optForce = list(optIFrcForAllSt[i].items())[0]
        #print('opt force for state : {:.5f}, {:.5f}'.format(optForce[0],optForce[1]))
        #set optimal force in env, and get new observation
        #NOTE : need to use setFrcMultFromFrc to set this so flags in env are set appropriately to use it and not a random force
        #env.wrapped_env.env.unwrapped.setFrcMultFromFrc(optForce[0],optForce[1])
        tFuncs.setFrcVals(env, optForce[0], optForce[1], False)
        #print('Calling use policy')
        path, path_baselines = tFuncs.usePolicy(env, trainDict, polDict, detPolicy=True, renderRes=False, debug=False)
        #add return for this solution
        returnsPerState.append(path['returns'][0])
        print('#{} : Const Force full return {:3.3f}'.format(i,path['returns'][0]))
    #env.wrapped_env.env.unwrapped.setDebugMode(False)
    
    #test with choosing every rollout force value via vf opt
    numRndFrc = 2
    for i in range(len(initQList)):
        #set initial state/state_dot
        env.wrapped_env.env.unwrapped.setNewState(initQList[i],initQdotList[i])   
        #setting initial force         
        _,optForce = list(optIFrcForAllSt[i].items())[0]
        #print('opt force for state : {:.5f}, {:.5f}'.format(optForce[0],optForce[1]))
        #set optimal force in env, and get new observation
        #NOTE : need to use setFrcMultFromFrc to set this so flags in env are set appropriately to use it and not a random force
        #env.wrapped_env.env.unwrapped.setFrcMultFromFrc(optForce[0],optForce[1])
        tFuncs.setFrcVals(env, optForce[0], optForce[1], False)
        path, path_baselines = tFuncs.usePolicyWithVFPreds(env, vfRD, trainDict, polDict, detPolicy=True, renderRes=False, numRndFrc=numRndFrc, optOnlyBad=False, debug=False)
        #add return for this solution
        returnsVFPerState.append(path['returns'][0])
        print('#{} : Opt Force Every Step full return {:3.3f}'.format(i,path['returns'][0]))
              
    #test with choosing initial and only bad force via opt
    numRndFrc = 10  #happens rarely so use better estimates
    for i in range(len(initQList)):
        #set initial state/state_dot
        env.wrapped_env.env.unwrapped.setNewState(initQList[i],initQdotList[i])            
        #setting initial force         
        _,optForce = list(optIFrcForAllSt[i].items())[0]
        #print('opt force for state : {:.5f}, {:.5f}'.format(optForce[0],optForce[1]))
        #set optimal force in env, and get new observation
        #NOTE : need to use setFrcMultFromFrc to set this so flags in env are set appropriately to use it and not a random force
        #env.wrapped_env.env.unwrapped.setFrcMultFromFrc(optForce[0],optForce[1])
        tFuncs.setFrcVals(env, optForce[0], optForce[1], False)
        path, path_baselines = tFuncs.usePolicyWithVFPreds(env, vfRD, trainDict, polDict, detPolicy=True, renderRes=False, numRndFrc=numRndFrc, optOnlyBad=True, debug=False)
        #add return for this solution
        returnsVFPerBad.append(path['returns'][0])
        print('#{} : Opt Force Init and Bad Rollout Steps full return {:3.3f}'.format(i,path['returns'][0]))
    
    
    for i in range(len(initQList)):
        print('#{} : Const Force rtn {:3.3f}|Opt Force per Step rtn {:3.3f}|Opt Force Bad Rollouts rtn {:3.3f}'.format(i,returnsPerState[i],returnsVFPerState[i],returnsVFPerBad[i]))
    #plot results
    xAxis = [x for x in range(len(initQList))] if idxsToUse is None else idxsToUse
    plt.plot(xAxis,returnsPerState,c='b', label='Init Step Only')
    plt.plot(xAxis,returnsVFPerState,c='r', label='Every Rollout Step')
    plt.plot(xAxis,returnsVFPerBad,c='g', label='Bad Rollout Steps')
    plt.legend(loc='lower left',ncol=4,fontsize=8)
    plt.title('Comparison of Returns Using VF Preds : Initial, Every Rollout Step, and Only Bad rollout steps\nfor {}'.format(tFuncs.getExpNameDisp(expDict)))
    plt.xlabel('Initial State IDX')
    plt.ylabel('Actual Return')
    plt.show()    

#build plots illustrating a heat map of return values for force x/y values
#using both value function and policy evalutionat for returns
def showForceRets(env, trainDict, polDict, expDict):
    #whether we should retrain the baseline with the results of the rollouts from saved states and show retrained bl results too
    retrainBL = False
    #initial state and state dot 
    #idxsToUse is a list of indexes in precalced state/statedot to use - set to none to use all
    #idxsToUse = None
    idxsToUse = [0]
    initQList, initQdotList = tFuncs.loadInitStates(env,expDict,idxsToUse=idxsToUse)
    #force values
    numVals = 21
    vList = np.linspace(0.0,0.5,num=numVals,endpoint=True)
    #vList = np.linspace(0.2,0.22,num=numVals,endpoint=True)
    frcVals = [(vList[x],vList[y]) for x in range(len(vList)) for y in range(len(vList))]

    #whether to use force value or force multiplier value - ignored for rwrd function calculation, only used to match training for baseline calc
    useFrc = expDict['bl_useForce']
    
    #rollout reward-based evaluation using original baseline and get all paths
    resListRwd, resDictListRwd, pathsRwd = execFunc(fitnessRwrd,env, trainDict, polDict, initQList, initQdotList, frcVals,useFrc)
    resMatRwd = np.transpose(-1*np.reshape(resListRwd,(numVals,numVals))) 
    #evaluate original baseline
    resListBLOrig, resDictListBLOrig, _ = execFunc(fitnessBL,env, trainDict, polDict, initQList, initQdotList, frcVals,useFrc) 
    resMatBLOrig = np.transpose(-1*np.reshape(resListBLOrig,(numVals,numVals)))
    
    #rollouts using value function predictions
#    resListRwd_VF, resDictListRwd_VF, pathsRwd_VF = execFunc(fitnessRwrdFrcFromVF,env, trainDict, polDict, initQList, initQdotList, frcVals,useFrc)
#    resMatRwd_VF = np.transpose(-1*np.reshape(resListRwd_VF,(numVals,numVals))) 
    
    
    numTtlPlots=2
    curPlot = 1
    tFuncs.HMPlot(vList, resMatBLOrig, 'Original Value Function Evaluations of Force Multipliers', curPlot,numTtlPlots=numTtlPlots)
    curPlot +=1    
    if retrainBL :
        numTtlPlots = 3
        #fit (retrain) baseline to collected paths using specified starting states - maybe try 2 iterations?
        polDict['baseline'].fit(pathsRwd)    
        #evaluate newly retrained baseline
        resListBLNew, resDictListBLNew, _ = execFunc(fitnessBL,env, trainDict, polDict, initQList, initQdotList, frcVals,useFrc) 
        resMatBLNew = np.transpose(-1*np.reshape(resListBLNew,(numVals,numVals)))
        
        tFuncs.HMPlot(vList, resMatBLNew, 'Retrained Value Function Evaluations of Force Multipliers', curPlot, numTtlPlots=3)
        curPlot +=1    
   
    tFuncs.HMPlot(vList, resMatRwd, 'Rollout Evaluations of Force Multipliers', curPlot,numTtlPlots=numTtlPlots)
    curPlot +=1    
#    tFuncs.HMPlot(vList, resMatRwd_VF, 'Rollout Evaluations Using VF Preds', curPlot,numTtlPlots=numTtlPlots)
#    curPlot +=1

#execute function passed with arguments and named arguments
def execFunc(func, *args, **kwargs):
    return func(*args, **kwargs)

#use CMA to find optimal force value for policy or baseline
def cmaOptFrcVals(env, trainDict, polDict, expDict):    
    import cma   
    #set to newPolDict if using retrained baseline
    polDictToUse = polDict
    #polDictToUse = newPolDict  #for trained baseline
    #use func==fitnessRwrd for policy, ==fitnessBL for baseline
    func = fitnessRwrd
    #func = fitnessBL
    
    #initial state and state dot for CMA optimization 
    #idxsToUse is a list of indexes in precalced state/statedot to use - set to none to use all
    idxsToUse = None
    #idxsToUse = [0]
    initQList, initQdotList = tFuncs.loadInitStates(env,expDict,idxsToUse=idxsToUse)
    
    #set CMA options
    opts = cma.CMAOptions()
    opts['bounds'] = [0, .5]  # Limit our values to be between 0 and .5
    opts['maxfevals'] = 450#max # of evals of fitness func
    opts['tolfun']=1.0e-02#termination criterion: tolerance in function value, quite useful
    #for each proposal set initial state/statedot
    #env.wrapped_env.env.unwrapped.setDebugMode(False)
    #initialize CMA optimizer
    initVals = 2 * [0.25]
    initSTD = 0.25
    es = cma.CMAEvolutionStrategy(initVals, initSTD, opts)
    itr = 1
    
    #whether to use force value or force multiplier value - ignored for rwrd function calculation, only used to match training for baseline calc
    useFrc = expDict['bl_useForce'] #should be true, not training policy using only multiplier
    
    #while not converged
    while not es.stop():
        solns = es.ask()
        #for each solution proposal, set init state/state-dot and sol proposal, record return
        resList, resDictList, _ = execFunc(func,env, trainDict, polDictToUse, initQList, initQdotList, solns,useFrc) 
        #inform CMA of results
        es.tell(solns,resList)
        es.logger.add() 
        es.disp()
        print('')
        print('iter : {} done'.format(itr))
        print('')
        itr = itr+1 
    es.result_pretty()
    cma.plot()  # shortcut for es.logger.plot()
    #result value : es.mean
    #actual force values, instead of multiplier
    frcX = env.wrapped_env.env.unwrapped.getForceFromMult(es.mean[0])
    frcY = env.wrapped_env.env.unwrapped.getForceFromMult(es.mean[1])
    print('final optimal force result : {},{}'.format(frcX, frcY))
    #for pose idx 0 using policy 'experiment_2017_12_17_08_06_21_0003'
    #final optimal force result : 38.44847564635957,91.39169846244793
    

#check baseline for reward given initial states and force proposals
#useFrc should reflect how policy was trained - whether force or force multiplier was used to train policy
#BASELINE NEEDS TO USE MODIFIED OBS FROM ENV,NOT SAVED STATE/STATE_dots
def fitnessBL(env, trainDict, polDict, initQList, initQdotList, solns, useFrc):
    solnRes = []
    solnDictList = []
    print('Force soln prop | Baseline res ')

    obsFromState = [env.wrapped_env.env.unwrapped.getObsFromState(initQList[x], initQdotList[x]) for x in range(len(initQList))]
    
    for s in solns :
        if((s[0]<0) or (s[1]<0) or (s[0]>.5) or(s[1]>.5)):
            print('WARNING :soln {} oob'.format(s))
        solnResDict = {}
        solnResDict['proposal'] = s
        retForSoln = 0
        if(useFrc):
            sFrc = np.array([env.wrapped_env.env.unwrapped.getForceFromMult(s[0]),env.wrapped_env.env.unwrapped.getForceFromMult(s[1])])
        else:
            sFrc = s        #observation variable changed to have only multiplier, not full assistive force value
        for i in range(len(initQList)):
            #set initial state/state_dot
            #'observations' needs to be a list of a list of observations - in this case a list of length 1 of list of initial state and force proposal
            #tmpPath = dict(observations=[np.concatenate((np.array(initQList[i][1:]),np.array(initQdotList[i]),sFrc))])
            #ignore assist force element of observation
            tmpPath = dict(observations=[np.concatenate((np.array(obsFromState[i][:-2]),sFrc))])
            path_baseline = polDict['baseline'].predict(tmpPath) 
            #add return for this solution
            retForSoln = retForSoln + path_baseline[0]
        #set results as negative returns
        solnResDict['eval'] = -retForSoln
        solnRes.append(solnResDict['eval'])
        solnDictList.append(solnResDict)
        print('{}|{:.4f}'.format(['{:.4f}'.format(i) for i in solnResDict['proposal']],solnResDict['eval']))
        #no paths so return None
    return solnRes, solnDictList, None


#check fitness of proposals - return list of dictionaries of results for all proposals
#useFrc ignored for reward calc
def fitnessRwrd(env, trainDict, polDict, initQList, initQdotList, solns, useFrc):
    solnRes = []
    solnDictList = []
    paths = []
    print('Force soln prop | Return res ')
    for s in solns :
        if((s[0]<0) or (s[1]<0) or (s[0]>.5) or(s[1]>.5)):
            print('WARNING :soln {} oob'.format(s))
        solnResDict = {}
        solnResDict['proposal'] = s
        #set force from solution set
        tFuncs.setFrcVals(env, s[0],s[1], True)
        #env.wrapped_env.env.unwrapped.setForceMag(s[0],s[1])
        retForSoln = 0
        #for every init state
        for i in range(len(initQList)):
            #set initial state/state_dot
            env.wrapped_env.env.unwrapped.setNewState(initQList[i],initQdotList[i])            
            path, path_baselines = tFuncs.usePolicy(env, trainDict, polDict, detPolicy=True, renderRes=False, debug=False)
            #add return for this solution
            retForSoln = retForSoln + path['returns'][0]
            #save path to paths for potential baseline retraining
            paths.append(path)
        #set results as negative returns
        solnResDict['eval'] = -retForSoln
        solnRes.append(solnResDict['eval'])
        solnDictList.append(solnResDict)
        print('{}|{:.4f}'.format(['{:.4f}'.format(i) for i in solnResDict['proposal']],solnResDict['eval']))
    return solnRes, solnDictList, paths

##check fitness of proposals - return list of dictionaries of results for all proposals
##useFrc ignored for reward calc
#def fitnessRwrdFrcFromVF(env, trainDict, polDict, initQList, initQdotList, solns, useFrc):
#    solnRes = []
#    solnDictList = []
#    paths = []
#    print('Force soln prop | Return res ')
#    for s in solns :
#        if((s[0]<0) or (s[1]<0) or (s[0]>.5) or(s[1]>.5)):
#            print('WARNING :soln {} oob'.format(s))
#        solnResDict = {}
#        solnResDict['proposal'] = s
#        #set force from solution set
#        env.wrapped_env.env.unwrapped.setForceMag(s[0],s[1])
#        retForSoln = 0
#        #for every init state
#        for i in range(len(initQList)):
#            #set initial state/state_dot
#            env.wrapped_env.env.unwrapped.setNewState(initQList[i],initQdotList[i])            
#            path, path_baselines = tFuncs.usePolicyWithVFPreds(env, vfRD, trainDict, polDict, detPolicy=True, renderRes=False, debug=False)
#            #add return for this solution
#            retForSoln = retForSoln + path['returns'][0]
#            #save path to paths for potential baseline retraining
#            paths.append(path)
#        #set results as negative returns
#        solnResDict['eval'] = -retForSoln
#        solnRes.append(solnResDict['eval'])
#        solnDictList.append(solnResDict)
#        print('{}|{:.4f}'.format(['{:.4f}'.format(i) for i in solnResDict['proposal']],solnResDict['eval']))
#    return solnRes, solnDictList, paths
#



##load particular set of initial states and state dots
##returns must always be in list of lists of q and list of lists of qdots
##idxsToUse : list of idxs to use, None means use all
#def loadInitStates(env,expDict, idxsToUse=None):
#    #initial state and state dot for CMA optimization
#    initQList, initQdotList = tFuncs.loadRndStatesForSkel(env,expDict['savedInitStates'])
#    #find optimal value for idx 0 of pre-calced init q and qdot
#    if(idxsToUse is not None):
#        initQList = [initQList[x] for x in idxsToUse]
#        initQdotList = [initQdotList[x] for x in idxsToUse]
#    return initQList, initQdotList
    


#old code               
#def findOptForceStaticAlpha(tmpPath, X, vf, vfJacob):
#    #copy of state/observation to use to optimize force components - format necessary for 
#    stateToOpt = list([np.array([x for x in tmpPath['observations'][0]])]) 
#    oldScore = 0
#    oldBestScore = 0
#    vfEval = [[0]]
#    i = 0
#    while (i<10):
#        i+= 1
#        #evaluate jacobian passed 
#        jacVal = vfJacob.eval({X:stateToOpt})
#        #print("gradient : delX={:5.5f} delY={:5.5f}".format(jacVal[0][0][-2],jacVal[0][0][-1])) 
#        alpha = .001           
#        #alpha = 20.0      
#        #adaptive step size
#        #while (alpha > .001) :
#        modVal1 = alpha * jacVal[0][0][-1]
#        modVal2 = alpha * jacVal[0][0][-2]
#        #reapply until gets worse
#        while True:
#            #copy last force values
#            #lastVals = stateToOpt[0][-2:]
#            #mod values
#            stateToOpt[0][-1] += modVal1
#            stateToOpt[0][-2] += modVal2
#            #copy old score as old best score
#            oldBestScore = oldScore
#            #copy previous iteration's score as old score
#            oldScore = vfEval[0][0]
#            #evaluate new state
#            vfEval = vf(stateToOpt)
#            #print('\t\tstate {:5.5f},{:5.5f} -> score:{} | oldScore:{}'.format(stateToOpt[0][-2],stateToOpt[0][-1],vfEval[0][0],oldScore))
#            if (vfEval[0] < oldScore):
#                #print('\t\t\talpha too big :{} '.format(alpha))
#                #print('\t\t too far : state {:5.5f},{:5.5f} score:{} | oldScore:{}'.format(stateToOpt[0][-2],stateToOpt[0][-1],vfEval[0][0],oldScore))
#                break
#            
#            #got worse so halve alpha and restore state and score
#            #alpha *= .5
#        stateToOpt[0][-1] -= modVal1
#        stateToOpt[0][-2] -= modVal2
#        vfEval = vf(stateToOpt)#vfEval[0] = oldScore
#        oldScore = oldBestScore
#        #print("\tBest state for alpha {:5.5f},{:5.5f} | alpha : {:4.4f} | old score {} : score {}".format(stateToOpt[0][-2],stateToOpt[0][-1], alpha,oldScore, vfEval[0]))            
#        #print("Best state for gradient : delX={:5.5f} delY={:5.5f} -> {:5.5f},{:5.5f} | alpha : {:4.4f} | old score {} : score {}".format(jacVal[0][0][-2],jacVal[0][0][-1],stateToOpt[0][-2],stateToOpt[0][-1], alpha,oldScore, vfEval[0][0]))            
#        #print("")
#    calcVfEval  = vfEval[0][0]
#    vfEval = vf(stateToOpt)
#    print("Best state :  {:5.5f},{:5.5f} | last score {} | calc score {}".format(stateToOpt[0][-2],stateToOpt[0][-1], calcVfEval, vfEval[0][0]))            
#          
#    return stateToOpt                
                        
#    for i in range(1000):
#        #evaluate jacobian passed 
#        jacVal = vfJacob.eval({X:stateToOpt})
#        #update with jacobian via gradient ascent repeatedly until not improving
#        repeatUpdate = True
#        alpha = 1.0
#        oldScore = [0]
#        #while score is increasing, reapply same jacobian only to force component
#        while repeatUpdate :
#            stateToOpt[0][-1] += alpha*jacVal[0][0][-1]
#            stateToOpt[0][-2] += alpha*jacVal[0][0][-2]
#            vfEval = vf(stateToOpt)
#            
#            if(vfEval <= oldScore):#if not improving
#                #remove bad update
#                stateToOpt[0][-1] -= alpha*jacVal[0][0][-1]
#                stateToOpt[0][-2] -= alpha*jacVal[0][0][-2]
#                #make alpha smaller and then repeat update
#                alpha *= .5  
#                #reset to old score to carry into next iteration
#                vfEval = oldScore#vf(stateToOpt)
#            else : 
#                print("delX={:5.5f} delY={:5.5f} -> {:5.5f},{:5.5f} | alpha : {:4.4f} | old score {} : score {}".format(jacVal[0][0][-2],jacVal[0][0][-1],stateToOpt[0][-2],stateToOpt[0][-1], alpha,oldScore[0], vfEval[0]))
#            #repeat with decreaseing alpha until below threshold
#            if(alpha < .01):
#                repeatUpdate = False
#            oldScore = vfEval
#       #print("delX={:5.5f} delY={:5.5f} -> {:5.5f},{:5.5f} : score {}".format(jacVal[0][0][-2],jacVal[0][0][-1],stateToOpt[0][-2],stateToOpt[0][-1], vfEval[0]))
#    return stateToOpt
    