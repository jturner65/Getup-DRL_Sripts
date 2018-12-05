#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 11:44:03 2017

@author: john
"""
#set working directory
#import os
#os.chdir('/home/john/rllab_project1/')
#these need to be added (at least temporarily) to path
import os, sys
projTestDir = os.path.expanduser( '~/rllab_project1/')
os.chdir(projTestDir)
sys.path.insert(0, projTestDir)

import trpoLibFuncs as tFuncs
from vfOptLibFuncs import vfOpt 
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#holds global value for simple/complx reward function
#import gym.envs.dart.dart_env_2bot as dartEnv2bot

dataDirName=None  #default

#haven't specified these yet for 3d bot
initStatesCSVName = None

#######################
#   3d kima w/KR5 arm

initStatesCSVName = 'Kima50InitStates.csv'
#3d env getting up with kinematic robot
#either Kima human or Biped human, either KR5 helper or Biped helper
#envName = 'DartStandUp3d_2Bot-v1'

#env to duplicate GAE paper results
#envName = 'DartStandUp3d_GAE-v1'

#larger network, larger net for baseline, partially trained - only around 560 iters
#dataDirName ='exp-011101110111_bl_64-64_pl_128-64-64_it_560_mxPth_500_nBtch_200000_ALG_TRPO__2018_09_10_13_32_21_0001'
#continuation of above - around 1361 iters
#dataDirName ='exp-011101110111_bl_64-64_pl_128-64-64_it_1361_mxPth_500_nBtch_200000_ALG_TRPO__2018_09_10_13_32_21_0001'
#still further trained - 1922 iters by here
#dataDirName ='exp-011101110111_bl_64-64_pl_128-64-64_it_1922_mxPth_500_nBtch_200000_ALG_TRPO__2018_09_10_13_32_21_0001'
#nearly max  - 3826
#dataDirName = 'exp-011101110111_bl_64-64_pl_128-64-64_it_3826_mxPth_500_nBtch_200000_ALG_TRPO__2018_09_10_13_32_21_0001'
#nearly max with small batch size
#dataDirName = 'exp-011101110111_bl_64-64_pl_128-64-64_it_17035_mxPth_500_nBtch_50000_ALG_TRPO__2018_09_24_14_14_47_0001'
#big policy/bl with fewer batches and (eventually) more iters
#dataDirName = 'exp-011101110111_bl_64-64_pl_128-64-64_it_2127_mxPth_500_nBtch_50000_ALG_TRPO__2018_09_24_14_14_47_0001'

#retrained big policy with expanded reward function to penalize cnstrnt frc and rwrd knee action
#dataDirName = 'exp-111011101110111_bl_64-64_pl_128-64-64_it_5432_mxPth_500_nBtch_200000_ALG_TRPO___2018_10_01_12_07_54_0001'
#dataDirName = 'exp-111011101110111_bl_64-64_pl_128-64-64_it_7700_mxPth_500_nBtch_200000_ALG_TRPO___2018_10_01_12_07_54_0001'

#training policy w/constraint and using constraint displacement as observation component
envName = 'DartStandUp3d_2Bot-v2'           #environment w/constraint using constraint disp as observation
#these policies trained constraint with relative location set to ANA's standing com (for some obscene reason that escapes me now, this has been removed)
#dataDirName ='exp-111011101110110_bl_32-32_pl_128-64-64_it_2090_mxPth_500_nBtch_200000_ALG_TRPO___2018_10_06_08_18_31_0001'
#retrained policy w/constraints with less emphasis on constraint frc minimization
#dataDirName = 'exp-111011101110110_bl_32-32_pl_128-64-64_it_3138_mxPth_500_nBtch_200000_ALG_TRPO___2018_10_08_08_05_06_0001'
#dataDirName = 'exp-111011101110110_bl_32-32_pl_128-64-64_it_3244_mxPth_500_nBtch_200000_ALG_TRPO___2018_10_08_08_05_06_0001'
#policy trained on env with 1/10 timestep and 10 frameskip
#dataDirName = 'exp-111011101110110_bl_32-32_pl_128-64-64_it_2754_mxPth_500_nBtch_200000_ALG_TRPO___2018_10_09_13_40_32_0001'
#further trained policy trained on 1/10 timestep and 10 frameskip
#dataDirName = 'exp-111011101110110_bl_32-32_pl_128-64-64_it_8886_mxPth_500_nBtch_200000_ALG_TRPO___2018_10_09_13_40_32_0001'

################################
# v2 env trained w/servo actuator root on traj component - These don't work well - knees don't straighten, penalizing constraint force doesn't have desired effect
#since servo body doesn't seem to generate appropriate constraint force
#dataDirName = 'exp-110011101110111_bl_32-32_pl_128-64-64_it_627_mxPth_500_nBtch_200000_ALG_TRPO___2018_11_01_13_39_55_0001'
#dataDirName = 'exp-110011101110111_bl_32-32_pl_128-64-64_it_4169_mxPth_500_nBtch_200000_ALG_TRPO___2018_11_01_13_39_55_0001'
#dataDirName = 'exp-110011101110111_bl_32-32_pl_128-64-64_it_5340_mxPth_500_nBtch_200000_ALG_TRPO___2018_11_01_13_39_55_0001'
####
# v2 env trained with servo actuator root but without any constraint force considerations
#dataDirName = 'exp-011011101110111_bl_32-32_pl_128-64-64_it_1710_mxPth_500_nBtch_200000_ALG_TRPO___2018_11_18_07_45_28_0001'
#dataDirName = 'exp-011011101110111_bl_32-32_pl_128-64-64_it_8586_mxPth_500_nBtch_200000_ALG_TRPO___2018_11_18_07_45_28_0001'
dataDirName = 'exp-011011101110111_bl_32-32_pl_128-64-64_it_13700_mxPth_500_nBtch_200000_ALG_TRPO___2018_11_18_07_45_28_0001'
#retraining attempt with smaller network archs - 1182 iters
#dataDirName = 'exp-011101110111_bl_32-32-16_pl_64-32-32_it_1182_mxPth_500_nBtch_200000_ALG_TRPO__2018_09_16_06_11_31_0001'
#2928 iters
#dataDirName = 'exp-011101110111_bl_32-32-16_pl_64-32-32_it_2928_mxPth_500_nBtch_200000_ALG_TRPO__2018_09_16_06_11_31_0001'
#nearly max : 4051 iters
#dataDirName = 'exp-011101110111_bl_32-32-16_pl_64-32-32_it_4051_mxPth_500_nBtch_200000_ALG_TRPO__2018_09_16_06_11_31_0001'

#fewer batches
#dataDirName ='exp-011101110111_bl_32-32-16_pl_64-32-32_it_4428_mxPth_500_nBtch_50000_ALG_TRPO__2018_09_20_13_47_57_0001'
#dataDirName ='exp-011101110111_bl_32-32-16_pl_64-32-32_it_5000_mxPth_500_nBtch_50000_ALG_TRPO__2018_09_20_13_47_57_0001'

#dataDirName=None, envName='DartStandUp3d_2Bot-v1', isNormalized=False, initStatesCSVName='testStateFile.csv',resumeTrain=True)
#set reward functions to use based on dataDirName - MUST BE CONFIGURED WITH 'exp-<xxxx>....' where <xxxx> is binary string of in-use reward components
dbgRwdsUsed=tFuncs.setEnvRwdFuncs(dataDirName)

dataDict = tFuncs.buildAllExpDicts(dataDirName=dataDirName, envName=envName, isNormalized=True, initStatesCSVName=initStatesCSVName, resumeTrain=True, useNewEnv=True)
expDict = dataDict['expDict']
polDict = dataDict['polDict']
trainDict = dataDict['trainDict']
dart_env=dataDict['dartEnv']


#setting reward components forcibly
#rwdList=['eefDist','action','height','lFootMovDist','rFootMovDist','comcop','UP_COMVEL','X_COMVEL','Z_COMVEL','kneeAction','matchGoalPose','assistFrcPen']
#to be able to modify global value for using simple reward
#dart_env.setDesiredRwdComps(rwdList)
#uncomment to start training in single-thread
#algo = tFuncs.trainPolicy(env, polDict, trainDict)

#uncomment to build and save a file of initial states to use
#_ = tFuncs.saveNumRndStatesForSkel(dart_env, expDict['savedInitStates'],numStates=50)
#contactDict, COPval, cntctVecDict, footAvgLoc = dataDict['dartEnv'].skelHldrs[dataDict['dartEnv'].humanIdx].calcFootContactData()
def cameraChange(dart_env, iters=200):
    #to change camera
    for i in range(iters):
        dart_env.render()  

cameraChange(polDict['env'],iters=500)
   
def test1RunPolicy(dataDict):    
    expDict = dataDict['expDict']        
    dart_env = dataDict['dartEnv']#derefed into inheriting env class
    assistDict = dataDict['assistDict']
    usePolicyArgsDict = defaultdict(int,{'pausePerStep':0, 'renderRes':1, 'recording':1, 'findBotFrc':0, 'iterSteps': 250,'rndPolicy':expDict['useRndPol'] })
    #dart_env.setAssistObjsCollidable()
    #dart_env.getANAHldr().calcCompensateEefFrc = True
    #
    #uncomment/run this line to save states
    dart_env.setStateSaving('ana',saveStates=True)#set to false to turn off - re-calling this will overwrite previous data, so don't do so unless overwrite is desired

    for i in range(100):
        #dart_env.setAssistDuringRollout( np.array([0.1, 0.1, 0.0]),assistDict) #smaller policy has very hard time with this, larger has easier time; Note : policies never saw this force during training
        path,_,_ = tFuncs.usePolicy(dataDict,optsDict=usePolicyArgsDict)
        #print('Force Mult Used : {} : frc : {}'.format(dart_env.frcMult, dart_env.assistForce))
        #run this line to update file with saved states
        dart_env.saveROStates('ana')    #if enabled state saving this will write to file. calls are ignored in skl hndlr otherwise
        #fileName=dart_env.getANAHldr().savedStateFilename
    #dart_env.frcMult
    
    
#run a single policy test run with bot assist where bot is actually applying assistance
def testBotPullAndVFOpt(dataDict):    
    expDict = dataDict['expDict']
    polDict = dataDict['polDict']
    
    dart_env = dataDict['dartEnv']#derefed into inheriting env class   
    dart_env.skelHldrs[dart_env.botIdx].debug=False
    usePolicyArgsDict = defaultdict(int,{'rndPolicy':expDict['useRndPol'], 'renderRes':1, 'recording':1, 'getEefFrcDicts':1, 'findBotFrc':0, 'iterSteps':200})
    assistInitSeedVal = np.array([0.01,0.01,0.0])
    
    #set up to use every step bot force gen : 1 is every step gets force update
    # number of random forces to explore for each initial state to find optimal result in value function
    #ignores assist force
    numRndVals=7 
    #get theano-derived net values and jacobians from baseline
    vfOptObj = vfOpt(polDict['baseline']._regressor, dart_env, numRndVals)
    assistMethod='Value Function Optimization'
    #dart_env.setCurrPolExpDict(polDict, expDict, newAssistVal, vfOptObj=None):)
    dart_env.setCurrPolExpDict(polDict, expDict, initAssistVal=assistInitSeedVal,vfOptObj=vfOptObj)
    
    #run policy rollout
    path,_, eefFrcResAra = tFuncs.usePolicy(dataDict,optsDict=usePolicyArgsDict)
    #show total torques for bot and ANA
    calcTtlTorques(eefFrcResAra, assistMethod)
    #build heatmap image ara
    
#build plots of results from rollout state-action-state' files
def dispSASResPlots(dataDict):

    dart_env = dataDict['dartEnv']#derefed into inheriting env class
    #this is current file name from current run of policy in env - may need to replace     
    #stateFileName=dart_env.getANAHldr().savedStateFilename
    #pre-existing stateFileName  
    #stateFileName='/home/john/rolloutStateSaves/ANA_ANA_:_Agent_Needing_Assistance_BOT_KR5_Helper_Arm_RWD_0111011101011_2018-09-17-11-39-52-518033/rolloutStates.csv'
    #stateFileName='/home/john/rolloutCSVData/ANA_RO_SASprime/ANA_ANA_:_Agent_Needing_Assistance_BOT_KR5_Helper_Arm_RWD_000011101110111_2018-10-01-07-49-59-424107/ANA_RO_SASprimeData.csv'
    tmpData = np.genfromtxt(stateFileName,skip_header=1, delimiter=',') 
    #dictionary containing idxs for components of SAS vector for ana
    fmtDict=dart_env.getANAHldr().SASVecFmt
    idxs = fmtDict['action']
    idxs = fmtDict['kneeDOFActIdxs']
    
    tmpActionData = tmpData[:,idxs]
    
    szAct = len(tmpActionData[0])+1
    actSlice = slice(0, szAct, 1)
    #test actions
    tFuncs.plotTestValsAvgStdMinMax(tmpActionData, actSlice, 'Policy Sampled Actions')


#this will display and plot all appropriate training information that is saved in progress.csv, written by expLite
#dictionary is loaded into expDict when it is built for all existing experiments
def dispTrainingInfo(dataDict):    
    expDict = dataDict['expDict']
    #keys : ['Entropy', 'dLoss', 'StdReturn', 'vf_dLoss', 'Perplexity', 'MinReturn', 'vf_LossBefore', 'LossBefore', 'NumTrajs', 'Iteration', 'LossAfter', 'MaxReturn', 'vf_LossAfter', 'vf_MeanKL', 'AverageReturn', 'AverageDiscountedReturn', 'ExplainedVariance', 'AveragePolicyStd', 'MeanKLBefore', 'MeanKL']
    keysToPlot=['AverageDiscountedReturn']
    keysToPlot2=['MinReturn','MaxReturn','AverageReturn']
    #build experiment description from directory name - only applicable to newer experiments that follow the trpoTests_ExpLite format
    expNameDict, expDesc, shortName = tFuncs.expDirNameToString(expDict['ovrd_dataDirName'])
    #tFuncs.plotPerfData(expDesc,expDict['trainingInfoDict'], keysToPlot, clipYAtZero=False)
    tFuncs.plotPerfData(expDesc,expDict['trainingInfoDict'], [keysToPlot,keysToPlot2],  yAxisScale='symlog',clipYAtZero=False)
    #linear y axis 
    
    keysToPlot1=['AveragePolicyStd']
    keysToPlot2=['ExplainedVariance']
    #keysToPlot3=['Entropy']
    keysToPlot = [keysToPlot1,keysToPlot2]
     #keysToPlot=['MinReturn','MaxReturn','AverageReturn']
    tFuncs.plotPerfData(expDesc, expDict['trainingInfoDict'], keysToPlot, clipYAtZero=True)
    #log y axis
    keysToPlot=['MinReturn','MaxReturn','AverageReturn']
    tFuncs.plotPerfData(expDesc, expDict['trainingInfoDict'], [keysToPlot], yAxisScale='log', clipYAtZero=True)
    #log y axis
    keysToPlot=['MinReturn','MaxReturn','AverageReturn']
    tFuncs.plotPerfData(expDesc, [keysToPlot], yAxisScale='symlog', clipYAtZero=True)
    #num trajectories
    keysToPlot=['NumTrajs']
    tFuncs.plotPerfData(expDesc, expDict['trainingInfoDict'], [keysToPlot], clipYAtZero=False)    
    
#compare 2 policies
def compPols():
    import os, sys
    rllabProjHome = os.path.expanduser( '~/rllab_project1/')
    sys.path.insert(0, rllabProjHome)
    os.chdir(rllabProjHome)
    import trpoLibFuncs as tFuncs
    
    keysToPlot=['AverageDiscountedReturn']
    keysToPlot2=['MinReturn','MaxReturn','AverageReturn']
    keysToPlotList = [keysToPlot, keysToPlot2]
    envName = 'DartStandUp3d_2Bot-v1'
    polDirAra=[
            'exp-011101110111_bl_64-64_pl_128-64-64_it_3826_mxPth_500_nBtch_200000_ALG_TRPO__2018_09_10_13_32_21_0001',
            'exp-011101110111_bl_64-64_pl_128-64-64_it_17035_mxPth_500_nBtch_50000_ALG_TRPO__2018_09_24_14_14_47_0001',
           # 'exp-011101110111_bl_32-32-16_pl_64-32-32_it_4051_mxPth_500_nBtch_200000_ALG_TRPO__2018_09_16_06_11_31_0001',
            #'exp-011101110111_bl_64-64_pl_128-64-64_it_2127_mxPth_500_nBtch_50000_ALG_TRPO__2018_09_24_14_14_47_0001'
           #'exp-011101110111_bl_32-32-16_pl_64-32-32_it_5000_mxPth_500_nBtch_50000_ALG_TRPO__2018_09_20_13_47_57_0001'
               ]
    
    normalizePerSmpl=True 
    yAxisScale='linear'
    clipYAtZero=True
    
    #tFuncs.plotPolsSideBySide(polDirAra, envName=envName, normEnv=True, keysToPlotList=keysToPlotList, yAxisScale='linear', clipYAtZero=True)
    expDictAra = tFuncs.plotPolsTogether(polDirAra, envName=envName, normEnv=True, normalizePerSmpl=normalizePerSmpl, keysToPlotList=keysToPlotList, yAxisScale=yAxisScale, clipYAtZero=clipYAtZero)
    
    
    

#show policy distributions for states from rollouts
def testPolicyOnROStates(dataDict):    
    expDict = dataDict['expDict']
    polDict = dataDict['polDict']
    #acquire rollout paths
    usePolicyArgsDict = defaultdict(int,{'rndPolicy':expDict['useRndPol'], 'renderRes':0, 'recording':0, 'findBotFrc':0, 'iterSteps': 1000 })
    paths = []
    for i in range(20):
        path,_,_ = tFuncs.usePolicy(dataDict,optsDict=usePolicyArgsDict)
        paths.append(path)

    resDict = tFuncs.testPolicyOnROStates(polDict, paths)
    obsLimits =  resDict['skelValLims']
    #colSlice = slice(6, dart_env.skelHldrs[dart_env.humanIdx].ndofs, 1)
    colSlice = slice(6, 2*dart_env.skelHldrs[dart_env.humanIdx].ndofs, 1)
    tFuncs.plotTestObsAndLims(resDict['obs'],obsLimits, colSlice)
    
    szAct = len(resDict['actions'][0])
    actSlice = slice(0, szAct, 1)
    #test actions
    tFuncs.plotTestValsAvgStdMinMax(resDict['actions'], actSlice, 'Policy Sampled Actions')
    
    #test mean actions
    tFuncs.plotTestValsAvgStdMinMax(resDict['meanActions'], actSlice, 'Policy Mean Actions', lims=[-3.0,3.0])
    
    #test mean actions from rollouts
    tFuncs.plotTestValsAvgStdMinMax(resDict['meanActionsRO'], actSlice, 'Policy Mean RO Actions', lims=[-3.0,3.0])
    
#    res['skelValLims']=skelValLims
#    res['obs']=np.array(ttlObs)
#    #actions sampled from pol dist at given state
#    res['actions']=np.array(ttlActions)
#    #mean of pol dist at given state
#    res['meanActions']=np.array(ttlMeanActions)
#    #mean actions used in rollout- have been "normalized"
#    res['meanActionsRO']=np.array(ttlActionsFromRO)
#    #number of states
#    res['numStatesTested']=numStatesToTest
#    #compare actions from deterministic policy queries - should -all- be equal
#    res['actionSimilarity']={'eqAction':eqAction, 'closeAction':closeAction, 'farAction':farAction}


#test passed policy in polDict on random states
def testPolicyOnRndStates(dataDict):   
    polDict = dataDict['polDict']
    dart_env = dataDict['dartEnv']
    resDict = tFuncs.testPolicyOnRandomStates(polDict, numStatesToTest=10000)
    obsLimits =  resDict['skelValLims']
    #colSlice = slice(6, dart_env.skelHldrs[dart_env.humanIdx].ndofs, 1)
    colSlice = slice(6, 2*dart_env.skelHldrs[dart_env.humanIdx].ndofs, 1)
    tFuncs.plotTestObsAndLims(resDict['obs'],obsLimits, colSlice)
    #dart_env.skelHldrs[dart_env.humanIdx].skel.com()
    szAct = len(resDict['actions'][0])
    actSlice = slice(0, szAct, 1)
    #test actions
    tFuncs.plotTestValsAvgStdMinMax(resDict['actions'], actSlice, 'Policy Sampled Actions')
    
    #test mean actions
    tFuncs.plotTestValsAvgStdMinMax(resDict['meanActions'], actSlice, 'Policy Mean Actions')
    
    dart_env.skelHldrs[dart_env.humanIdx].dbgShowDofLims()


#run a single policy test run with bot assist force where bot is actually applying force
def testBotPullAndVFOpt_frc(dataDict):    
    expDict = dataDict['expDict']
    polDict = dataDict['polDict']
    useConstFrc = False
    
    dart_env = dataDict['dartEnv']#derefed into inheriting env class   
    dart_env.skelHldrs[dart_env.botIdx].debug=False
    usePolicyArgsDict = defaultdict(int,{'rndPolicy':expDict['useRndPol'], 'renderRes':1, 'recording':0, 'getEefFrcDicts':1, 'findBotFrc':0, 'iterSteps':200})
    assistInitSeedVal = np.array([0.01,0.01,0.0])
    
    if useConstFrc :
        frcMethod = 'Const Force Mult : {}'.format(assistInitSeedVal)
        #dart_env.setCurrPolExpDict(polDict, expDict,frcMult=assistFrcMultCnst, updateAction=2)
        dart_env.setCurrPolExpDict(polDict, expDict,initAssistVal=assistInitSeedVal)

    else:
        #set up to use every step bot force gen : 1 is every step gets force update
        # number of random forces to explore for each initial state to find optimal result in value function
        #ignores assist force
        numRndVals=7 
        #get theano-derived net values and jacobians from baseline
        vfOptObj = vfOpt(polDict['baseline']._regressor, dart_env, numRndVals)
        frcMethod='Value Function Optimization'
        #dart_env.setCurrPolExpDict(polDict, expDict, newAssistVal, vfOptObj=None):)
        dart_env.setCurrPolExpDict(polDict, expDict, initAssistVal=assistInitSeedVal,vfOptObj=vfOptObj)
    
    #run policy rollout
    path,_, eefFrcResAra = tFuncs.usePolicy(dataDict,optsDict=usePolicyArgsDict)
    #show total torques for bot and ANA
    calcTtlTorques(eefFrcResAra, frcMethod)
    #build heatmap image ara
    
    
#calculate the total torques expended by ana and bot during rollout using data in eefFrcResAra
def calcTtlTorques(eefFrcResAra, frcMethod):
    #process eefFrcResAra - array of per step dictionaries of per-frame tuples of frc results at end effector, along with torques, etc
    totAnaTrq=0
    totBotTrq=0
    iters =0
    for step in eefFrcResAra:
        for fr in step:
            iters+=1
            totAnaTrq += fr['ana']['tauMag']
            totBotTrq += fr['bot']['tauMag']
    print('Using {} to determine force yields ANA ttl torques : {} and bot ttl torques : {} over {} iterations'.format(frcMethod, totAnaTrq, totBotTrq, iters))
    
    #clip : Using Const Force Mult : [ 0.01  0.01  0.  ] to determine force yields ANA ttl torques : 187816.30874620005 and bot ttl torques : 17029.679174111367 over 840 iterations
    #Using Const Force Mult : [ 0.01  0.01  0.  ] to determine force yields ANA ttl torques : 177560.34162843635 and bot ttl torques : 16127.380147204469 over 840 iterations
    #Using Const Force Mult : [ 0.01  0.01  0.  ] to determine force yields ANA ttl torques : 188006.10981632915 and bot ttl torques : 18533.001098727542 over 888 iterations
    
    #clip : Using Const Force Mult : [ 0.5  0.5  0. ] to determine force yields ANA ttl torques : 169612.7750223577 and bot ttl torques : 217492.9932874536 over 840 iterations 
    #Using Const Force Mult : [ 0.5  0.5  0. ] to determine force yields ANA ttl torques : 169765.82278412298 and bot ttl torques : 217045.17928216647 over 840 iterations   
    #Using Const Force Mult : [ 0.5  0.5  0. ] to determine force yields ANA ttl torques : 180500.17808162278 and bot ttl torques : 237858.01692147847 over 888 iterations
    
    #clip : Using Value Function Optimization to determine force yields ANA ttl torques : 190209.5967072195 and bot ttl torques : 65566.98355655403 over 840 iterations

    #Using Value Function Optimization to determine force yields ANA ttl torques : 192447.04571242776 and bot ttl torques : 60287.346657189664 over 840 iterations    
    #Using Value Function Optimization to determine force yields ANA ttl torques : 192561.31928280432 and bot ttl torques : 61098.966561806104 over 840 iterations
    #Using Value Function Optimization to determine force yields ANA ttl torques : 208412.86370929654 and bot ttl torques : 59838.39302058543 over 888 iterations
    
    #Using Const Force Mult : [ 0.01  0.01  0.  ] to determine force yields ANA ttl torques : 177560.34162843635 and bot ttl torques : 16127.380147204469 over 840 iterations
    #Using Const Force Mult : [ 0.5  0.5  0. ] to determine force yields ANA ttl torques : 169765.82278412298 and bot ttl torques : 217045.17928216647 over 840 iterations   
    #Using Value Function Optimization to determine force yields ANA ttl torques : 182561.31928280432 and bot ttl torques : 61098.966561806104 over 840 iterations


#this function will take a sequence of states and a value function and will save a sequence
#of heat map images, one per state, that illustrate each state's performance in assist force multiplier domain (x, y w/z ==0)
def buildHeatMapVid(dataDict, eefFrcResAra):
    expDict = dataDict['expDict']  
    
    
    dirName = expDict['ovrd_dataDirName']+'_opt'
    statesAra = [ fr['ana']['state'] for s in eefFrcResAra for fr in s]    
    
    directory = os.path.join(os.path.expanduser( '~/vfOpt_heatMap/') + dirName)
    if not os.path.exists(directory):
        os.makedirs(directory)
    imgFN_prefix = os.path.join(directory,'heatMap')    
    #force values
    numVals = 11
    vList = np.linspace(0.0,0.5,num=numVals,endpoint=True)
    frcVals = [np.array([vList[x],vList[y], 0.0]) for x in range(len(vList)) for y in range(len(vList))]
    #whether to use force value or force multiplier value - ignored for rwrd function calculation, only used to match training for baseline calc
    useMultNotForce = expDict['bl_useMultNotForce'] #should be false, not training policy using only multiplier - set in environment ctor ONLY
    
    for i in range(len(statesAra)):
        st = [statesAra[i]]#needs to be a list of 1 element for fitnessBL function
        fileName = imgFN_prefix + '_{:03}.png'.format(i)
        
        resListBLOrig, resDictListBLOrig, _ = execFunc(fitnessBL_obs, dataDict, st, frcVals,useMultNotForce) 
        resMatBLOrig = np.transpose(-1*np.reshape(resListBLOrig,(numVals,numVals)))
    
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        plt.title('Original Value Function Evaluations of Force Multipliers for frame {}'.format(i))
        plt.pcolor(vList,vList,resMatBLOrig)
        plt.colorbar()
        fig.savefig(fileName)   # save the figure to file
        plt.close(fig)    # close the figure    
        
            
#run this to clear values set to handle vf optimization and perstep assignment of forces to bot
def clearDartEnvForNormalUse(dataDict):    
    expDict = dataDict['expDict']
    dart_env = dataDict['dartEnv']#derefed into inheriting env class   
    #clear out connections, set updateAction to 0
    dart_env.clearCurrPolExpDict()
    usePolicyArgsDict = defaultdict(int,{'rndPolicy':expDict['useRndPol'], 'renderRes':1, 'recording':0, 'findBotFrc':0, 'iterSteps':150})
    path,_,_ = tFuncs.usePolicy(dataDict,optsDict=usePolicyArgsDict)


#direct access testing
def testPolicy(dataDict):    
    expDict = dataDict['expDict']
    polDict = dataDict['polDict']
    usePolicyArgsDict = defaultdict(int,{'rndPolicy':expDict['useRndPol'], 'renderRes':1, 'recording':1, 'findBotFrc':0, 'iterSteps': 500 })
    for i in range(1000):
        
        #dart_env.setDebugMode(True)
        #dart_env.dart_world.dt 
        #dart_env.skelHldrs[dart_env.humanIdx].
        #anaHldr = dart_env.skelHldrs[dart_env.humanIdx]
        #botHldr = dart_env.skelHldrs[dart_env.botIdx]
        #anaHldr.lclCnstLoc
        #anaHldr.skel.body(anaHldr.reach_hand).to_local(botHldr.skel.body(botHldr.reach_hand).to_world(botHldr.lclCnstLoc))
        #anaHldr.dbg_getEffLocWorld()
        #botHldr.dbg_getEffLocWorld()
        #dart_env.trackTraj.trackObj.com()
        path,_,_ = tFuncs.usePolicy(dataDict,optsDict=usePolicyArgsDict)
        #dart_env.assistForce
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
    #           fontsize=8)ue
    plt.title('Comparison of Actual Returns and Value Function Predictions\nfor {}'.format(tFuncs.getExpNameDisp(expDict)))
    plt.xlabel('Rollout Step')
    plt.ylabel('Return')
    plt.show()


#read in weights of trained baseline, build duplicate network based on mean network
#net is value function network, valueFunc is theano function to call with state to get value
#pass regressor from trained baseline
def testBLPredsInPolicy(dataDict):    
    expDict = dataDict['expDict']
    polDict = dataDict['polDict']
    #for ordered dict
    import collections
    dart_env = dataDict['dartEnv']#derefed into inheriting env class
    usePolicyArgsDict = defaultdict(int,{'rndPolicy':expDict['useRndPol'], 'renderRes':1, 'recording':0, 'findBotFrc':0})
    #usePolicyArgsDict = {'rndPolicy':expDict['useRndPol'], 'renderRes':True, 'recording':False, 'debug':False, 'optOnlyBad':True}
    assistDict = dataDict['assistDict']

    #load predetermined q and qdot as init states
    #idxsToUse is a list of indexes in precalced state/statedot to use - set to none to use all
    #idxsToUse = None
    idxsToUse = [0]
    #poorly performing initial states
    #idxsToUse = [13,22,28,30,43,44]

    initQList, initQdotList = tFuncs.loadInitStates(dart_env,expDict,idxsToUse=idxsToUse)    

    # number of random forces to explore for each initial state to find optimal result in value function
    numRndAssist=50
    #get theano-derived net values and jacobians from baseline
    vfOptObj = vfOpt(polDict['baseline']._regressor, dart_env, numRndAssist)
    #results for all initial states being used - list of each state's resDict sorted dictionary of optimal forces and VF-predicted scores
    vfResAllStates = getOptFrcForAllStates(dart_env,vfOptObj, idxsToUse, initQList,initQdotList)
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
            dart_env.setAssistDuringRollout(v, assistDict)

            dart_env.setNewState(q,qdot)    
            #perform rollout
            path, path_baselines,_ = tFuncs.usePolicy(dataDict,optsDict=usePolicyArgsDict)
            #get return for this initial state and force value
            resRO[k]=path['returns'][0]
            obs = dart_env._get_obs()
            _,obsFrc,_ = dart_env.getObsComponents(obs)
            print('Rollout : Obs Force : {} | fed force :  {}}| return: {} '.format(obsFrc,v,resRO[k]))
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
            print('Force : {} |  VF pred return = {:5.5f} | Rollout Act Return = {:5.5f}'.format(s, k, v))
        

#return a list of optimal force dictionaries for all states passed
def getOptFrcForAllStates(dart_env, vfOptObj, idxsToUse, initQList,initQdotList, debug=False):
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
        #build observation from state
        obs = dart_env.getObsFromState(q,qdot)
        resDict = vfOptObj.findVFOptAssistForObs(obs)        
        if(debug):
            for k,v in resDict.items():
                print('For state IDX {} :: best VF Pred score : {} | force : {}'.format(stIdx,k,v))
        #optimal force according to value function : 
        #score,optForce = list(resDict.items())[0]
        vfResAllStates.append(resDict) 
        
    return vfResAllStates
        
#retrain baseline with select examples to attempt to improve performance
def retrainBLWithExamples(dataDict):    
    expDict = dataDict['expDict']
    polDict = dataDict['polDict']
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
    #iterations to retrain policy (# of runs of "fit" - should only need 1?)
    rtrnDict['retrainIters'] = 5
    #whether or not using deterministic policy - want to use random policy to enable bad rollouts
    rtrnDict['rndPolicy'] = True  #expDict['useRndPol']
    #draw results
    rtrnDict['renderRes'] = False
    #setAssist restricts force used per iter to be in region of CMA-predicted best choice
    rtrnDict['setAssist'] = True
    #bounds of force to set - TODO change to list of values?
    rtrnDict['forceToSet'] = ()
    #dictionary describing assistance values
    assistDict = dataDict['assistDict']

    dataDict['polDict']=newPolDict
    newBaseLine, paths, path_bls = tFuncs.retrainBaseline(dataDict, rtrnDict, assistDict)
    newPolDict['baseline']=newBaseLine
    #updated baseline
    tFuncs.testMaxBLAgainstRet(dataDict)
    #orig base line
    dataDict['polDict']=polDict
    tFuncs.testMaxBLAgainstRet(dataDict)    

#direct access testing
def testPolicyOnInitStates(dataDict):    
    expDict = dataDict['expDict']
    polDict = dataDict['polDict']
    dart_env = dataDict['dartEnv']
    usePolicyArgsDict = defaultdict(int,{'rndPolicy':expDict['useRndPol'], 'renderRes':1, 'recording':0, 'findBotFrc':0})
    #usePolicyArgsDict = {'rndPolicy':expDict['useRndPol'], 'renderRes':True, 'recording':False, 'debug':False, 'optOnlyBad':True}

    #initial state and state dot 
    #idxsToUse is a list of indexes in precalced state/statedot to use - set to none to use all
    idxsToUse = None
    #idxsToUse = [0]
    initQList, initQdotList = tFuncs.loadInitStates(dart_env,expDict,idxsToUse=idxsToUse)
    
    for i in range(len(initQList)):
        dart_env.setNewState(initQList[i],initQdotList[i])  
        #dart_env.setDebugMode(True)
        path,_,_ = tFuncs.usePolicy(dataDict, optsDict=usePolicyArgsDict)
        #dart_env.frcOffset
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
def showVFPerfForAllStates(dataDict):    
    expDict = dataDict['expDict']
    polDict = dataDict['polDict']
    dart_env = dataDict['dartEnv']#derefed into inheriting env class
    usePolicyArgsDict = defaultdict(int)
    #usePolicyArgsDict = {'rndPolicy':False, 'renderRes':False, 'recording':False, 'debug':False, 'optOnlyBad':True}
    assistDict = dataDict['assistDict']

    #initial state and state dot 
    #idxsToUse is a list of indexes in precalced state/statedot to use - set to none to use all
    idxsToUse = None
    #idxsToUse = [0]
    #poorly performing nitial states
    #idxsToUse = [13,22,28,30,43,44]
    initQList, initQdotList = tFuncs.loadInitStates(dart_env,expDict,idxsToUse=idxsToUse)
    # number of random forces to explore for each state to find optimal result in value function
    numRndAssist=20
    
    vfOptObj = vfOpt(polDict['baseline']._regressor, dart_env,numRndAssist)

    #for consistency - get same force for each state to use as initial guess
    optIFrcForAllSt = getOptFrcForAllStates(dart_env,vfOptObj, idxsToUse, initQList,initQdotList, numRndAssist)

    returnsPerState = []
    returnsVFPerState = []
    returnsVFPerBad = []
    #test with choosing only initial force
    #dart_env.setDebugMode(True)
    for i in range(len(initQList)):
        #set initial state/state_dot
        dart_env.setNewState(initQList[i],initQdotList[i]) 
        #set optimal initial force given state
        _,optForce = list(optIFrcForAllSt[i].items())[0]
        #print('opt force for state : {}'.format(optForce))
        #set optimal force in env, and get new observation
        #NOTE : need to use setAssistDuringRollout to set this so flags in env are set appropriately to use it and not a random force
        dart_env.setAssistDuringRollout(optForce, assistDict)

        #print('Calling use policy')
        path, path_baselines,_ = tFuncs.usePolicy(dataDict, optsDict=usePolicyArgsDict)
        #add return for this solution
        returnsPerState.append(path['returns'][0])
        print('#{} : Const Force full return {:3.3f}'.format(i,path['returns'][0]))
    #dart_env.setDebugMode(False)
    
    #test with choosing every rollout force value via vf opt
    #usePolicyArgsDict['optOnlyBad']=False
    numRndAssist = 3
    for i in range(len(initQList)):
        #set initial state/state_dot
        dart_env.setNewState(initQList[i],initQdotList[i])   
        #setting initial force         
        _,optForce = list(optIFrcForAllSt[i].items())[0]
        #print('opt force for state : {}'.format(optForce))
        #set optimal force in env, and get new observation
        #NOTE : need to use setAssistDuringRollout to set this so flags in env are set appropriately to use it and not a random force
        dart_env.setAssistDuringRollout(optForce, assistDict)

        path, path_baselines,_ = tFuncs.usePolicy(dataDict,vfOptObj=vfOptObj,numRndAssist=numRndAssist, optsDict=usePolicyArgsDict)
        #add return for this solution
        returnsVFPerState.append(path['returns'][0])
        print('#{} : Opt Force Every Step full return {:3.3f}'.format(i,path['returns'][0]))
              
    #test with choosing initial and only bad force via opt
    usePolicyArgsDict['optOnlyBad']=1
    numRndAssist = 10  #happens rarely so use better estimates
    for i in range(len(initQList)):
        #set initial state/state_dot
        dart_env.setNewState(initQList[i],initQdotList[i])            
        #setting initial force         
        _,optForce = list(optIFrcForAllSt[i].items())[0]
        #print('opt force for state : {}'.format(optForce))
        #set optimal force in env, and get new observation
        #NOTE : need to use setAssistDuringRollout to set this so flags in env are set appropriately to use it and not a random force
        dart_env.setAssistDuringRollout(optForce, assistDict)

        path, path_baselines,_ = tFuncs.usePolicy(dataDict,vfOptObj=vfOptObj, numRndAssist=numRndAssist, optsDict=usePolicyArgsDict)
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
#using both value function and policy evalution for returns
def showForceRets(dataDict):    
    expDict = dataDict['expDict']
    polDict = dataDict['polDict']
    trainDict = dataDict['trainDict']
    dart_env = dataDict['dartEnv']#derefed into inheriting env class
    usePolicyArgsDict = defaultdict(int)
    #usePolicyArgsDict = {'rndPolicy':False, 'renderRes':False, 'recording':False, 'debug':False, 'optOnlyBad':True}

    #whether we should retrain the baseline with the results of the rollouts from saved states and show retrained bl results too
    retrainBL = False
    #initial state and state dot 
    #idxsToUse is a list of indexes in precalced state/statedot to use - set to none to use all
    idxsToUse = None
    #idxsToUse = [0]
    initQList, initQdotList = tFuncs.loadInitStates(dart_env,expDict,idxsToUse=idxsToUse)
    #force values
    numVals = 21
    vList = np.linspace(0.0,0.5,num=numVals,endpoint=True)
    #vList = np.linspace(0.2,0.22,num=numVals,endpoint=True)
    #holding z fixed at 0
    frcVals = [np.array([vList[x],vList[y], 0.0]) for x in range(len(vList)) for y in range(len(vList))]

    #whether to use force value or force multiplier value - ignored for rwrd function calculation, only used to match training for baseline calc
    useMultNotForce = expDict['bl_useMultNotForce'] #should be false, not training policy using only multiplier - set in environment ctor ONLY
    
    #rollout reward-based evaluation using original baseline and get all paths
    resListRwd, resDictListRwd, pathsRwd = execFunc(fitnessRwrd,dataDict, initQList, initQdotList, frcVals,useMultNotForce)
    resMatRwd = np.transpose(-1*np.reshape(resListRwd,(numVals,numVals))) 
    #evaluate original baseline
    resListBLOrig, resDictListBLOrig, _ = execFunc(fitnessBL, dataDict, initQList, initQdotList, frcVals,useMultNotForce) 
    resMatBLOrig = np.transpose(-1*np.reshape(resListBLOrig,(numVals,numVals)))
    
    #rollouts using value function predictions
#    resListRwd_VF, resDictListRwd_VF, pathsRwd_VF = execFunc(fitnessRwrdFrcFromVF, dataDict, initQList, initQdotList, frcVals,useMultNotForce)
#    resMatRwd_VF = np.transpose(-1*np.reshape(resListRwd_VF,(numVals,numVals)))     
    
    numTtlPlots=1
    curPlot = 1
    tFuncs.HMPlot(vList, resMatBLOrig, 'Original Value Function Evaluations of Force Multipliers', curPlot,numTtlPlots=numTtlPlots)
    curPlot +=1    
    if retrainBL :
        numTtlPlots = 3
        #fit (retrain) baseline to collected paths using specified starting states - maybe try 2 iterations?
        polDict['baseline'].fit(pathsRwd)    
        #evaluate newly retrained baseline
        resListBLNew, resDictListBLNew, _ = execFunc(fitnessBL,trainDict, polDict, initQList, initQdotList, frcVals,useMultNotForce) 
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
def cmaOptFrcVals(dataDict):    
    expDict = dataDict['expDict']
    polDict = dataDict['polDict']
    dart_env = dataDict['dartEnv']#derefed into inheriting env class    
    import cma  
    #set to newPolDict if using retrained baseline
    polDictToUse = polDict
    #polDictToUse = newPolDict  #for trained baseline
    
    dataDict['polDict'] = polDictToUse
    #use func==fitnessRwrd for policy, ==fitnessBL for baseline
    func = fitnessRwrd
    #func = fitnessBL
    
    #initial state and state dot for CMA optimization 
    #idxsToUse is a list of indexes in precalced state/statedot to use - set to none to use all
    idxsToUse = None
    #idxsToUse = [0]
    initQList, initQdotList = tFuncs.loadInitStates(dart_env,expDict,idxsToUse=idxsToUse)
    
    #set CMA options
    opts = cma.CMAOptions()
    opts['bounds'] = [0, .5]  # Limit our values to be between 0 and .5
    opts['maxfevals'] = 450#max # of evals of fitness func
    opts['tolfun']=1.0e-02#termination criterion: tolerance in function value, quite useful
    #for each proposal set initial state/statedot
    #dart_env.setDebugMode(False)
    #initialize CMA optimizer
    initVals = dart_env.extAssistSize * [0.25]
    initSTD = 0.25
    es = cma.CMAEvolutionStrategy(initVals, initSTD, opts)
    itr = 1
    
    #whether to use force value or force multiplier value - ignored for rwrd function calculation, only used to match training for baseline calc
    useMultNotForce = expDict['bl_useMultNotForce'] #determined by environment    
    #while not converged
    while not es.stop():
        solns = es.ask()
        #for each solution proposal, set init state/state-dot and sol proposal, record return
        resList, resDictList, _ = execFunc(func,dataDict, initQList, initQdotList, solns,useMultNotForce) 
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
    frcDims = len(dart_env.frcMult)
    #actual assist values, instead of multipliers
    frcComp = dart_env.getForceFromMult(es.mean[0:-frcDims])
    print('final optimal force result : {}'.format(frcComp))
    if (dart_env.extAssistSize > frcDims):
        #has location component as well
        locComp = es.mean[-frcDims:]
        print('final optimal location of application result : {}'.format(locComp))
    #for pose idx 0 using policy 'experiment_2017_12_17_08_06_21_0003'
    #final optimal force result : 38.44847564635957,91.39169846244793
    

#check baseline for reward given initial states and force proposals
#useMultNotForce should reflect how policy was trained - whether force or force multiplier was used to train policy
#BASELINE NEEDS TO USE MODIFIED OBS FROM ENV,NOT SAVED STATE/STATE_dots
def fitnessBL(dataDict,initQList, initQdotList, cmaSolns, useMultNotForce):
    dart_env=dataDict['polDict']['dartEnv']
    obsFromState = [dart_env.getObsFromState(initQList[x], initQdotList[x]) for x in range(len(initQList))]
    return fitnessBL_obs(dataDict,obsFromState, cmaSolns, useMultNotForce)
#    lenFrcCmp = len(cmaSolns[0])
#    
#    obsPrefixList = [o[:-lenFrcCmp] for o in obsFromState]
#    #cmaSolns are force multipliers
#    for s in cmaSolns :
#        isLegal, dbgReason = dart_env.isLegalAssistVal(s, isFrcMult=True)
#        if not isLegal:
#            print('WARNING :soln {} bad : Reason : {}'.format(s, dbgReason))
#        solnResDict = {}
#        solnResDict['proposal'] = s
#        retForSoln = 0
#        if(useMultNotForce):
#            sFrc = s        #observation variable changed to have only multiplier, not full assistive force value
#        else:
#            sFrc = dart_env.getForceFromMult(s)
#        for obsPrefix in obsPrefixList:#for every state observation without the assist force
#            #set initial state/state_dot
#            #'observations' needs to be a list of a list of observations - in this case a list of length 1 of list of initial state and force proposal
#            #tmpPath = dict(observations=[np.concatenate((np.array(initQList[i][1:]),np.array(initQdotList[i]),sFrc))])
#            #ignore assist force element of observation
#            tmpPath = dict(observations=[np.concatenate((obsPrefix,sFrc))])
#            path_baseline = polDict['baseline'].predict(tmpPath) 
#            #add return for this solution
#            retForSoln = retForSoln + path_baseline[0]
#        #set results as negative returns
#        solnResDict['eval'] = -retForSoln
#        solnRes.append(solnResDict['eval'])
#        solnDictList.append(solnResDict)
#        print('{}|{:.4f}'.format(['{:.4f}'.format(i) for i in solnResDict['proposal']],solnResDict['eval']))
#        #no paths so return None
#    return solnRes, solnDictList, None

#same but using array of entire observation
def fitnessBL_obs(dataDict,obsFromState, cmaSolns, useMultNotForce):
    dart_env = dataDict['dartEnv']#derefed into inheriting env class
    solnRes = []
    solnDictList = []

    lenFrcCmp = len(cmaSolns[0])
    obsPrefixList = [o[:-lenFrcCmp] for o in obsFromState]
    #cmaSolns are force multipliers
    for s in cmaSolns :
        isLegal, dbgReason = dart_env.isLegalAssistVal(s, isFrcMult=True)
        if not isLegal:
            print('WARNING :soln {} bad : Reason : {}'.format(s, dbgReason))
        solnResDict = {}
        solnResDict['proposal'] = s
        retForSoln = 0
        if(useMultNotForce):
            sFrc = s        #observation variable changed to have only multiplier, not full assistive force value
        else:
            sFrc = dart_env.getForceFromMult(s)
        for obsPrefix in obsPrefixList:#for every state observation without the assist force
            #set initial state/state_dot
            #'observations' needs to be a list of a list of observations - in this case a list of length 1 of list of initial state and force proposal
            #tmpPath = dict(observations=[np.concatenate((np.array(initQList[i][1:]),np.array(initQdotList[i]),sFrc))])
            #ignore assist force element of observation
            tmpPath = dict(observations=[np.concatenate((obsPrefix,sFrc))])
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
#useMultNotForce ignored for reward calc
def fitnessRwrd(dataDict, initQList, initQdotList, cmaSolns, useMultNotForce):
    dart_env=dataDict['dartEnv']
    usePolicyArgsDict = defaultdict(int)
    #usePolicyArgsDict = {'rndPolicy':0, 'renderRes':0, 'recording':0, 'debug':0, 'optOnlyBad':1, 'findBotFrc':0}
    assistDict = dataDict['assistDict']
    solnRes = []
    solnDictList = []
    paths = []
    print('Force soln prop | Return res ')
    #cmaSolns are multipliers not force vals
    for s in cmaSolns :
        isLegal, dbgReason = dart_env.isLegalAssistVal(s, isFrcMult=useMultNotForce)
        if not isLegal:
            print('WARNING :soln {} bad : Reason : {}'.format(s, dbgReason))
        solnResDict = {}
        solnResDict['proposal'] = s
        #set force from solution set
        dart_env.setAssistDuringRollout(s, assistDict)
        retForSoln = 0
        #for every init state
        for i in range(len(initQList)):
            #set initial state/state_dot
            dart_env.setNewState(initQList[i],initQdotList[i])            
            path, path_baselines,_ = tFuncs.usePolicy(dataDict, optsDict=usePolicyArgsDict)
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

######### old dir data names    
#dataDirName = 'experiment_2018_03_29_18_24_04_0001'
#dataDirName = 'experiment_2018_03_24_18_48_36_0001'
#dataDirName = 'experiment_2018_03_24_06_54_00_0001'
#dataDirName = 'experiment_2018_03_23_20_46_16_0001'
#dataDirName = 'experiment_2018_03_23_14_21_24_0001'

#bad : dataDirName = 'experiment_2018_03_23_05_29_46_0001'
#dataDirName = 'experiment_2018_03_22_21_06_11_0001'
#dataDirName = 'experiment_2018_03_22_15_25_25_0001'
#dataDirName = 'experiment_2018_03_21_22_07_57_0001'
#dataDirName = 'experiment_2018_03_21_16_48_16_0001'
#below climbs up bot :
#dataDirName = 'experiment_2018_03_16_14_31_05_0001'

#2d walker getting up with assist force
#envName = 'DartStandUp2d-v1'
#these are all for env name DartStandUp2d-v1
#using iterative BL fitting
#dataDirName = 'experiment_2017_12_04_16_09_23_0001'
#using iterative BL fitting on bl with arch  : 16-16-8
#dataDirName = 'experiment_2017_12_04_21_13_05_0001'     
#name of csv file holding pretrained states
#initStatesCSVName='testStateFile.csv' 
#initStatesCSVName='testStateFile2.csv' 
#below was demonstrated to work using opt force suggestion - most demoed as of 2/1/18
#dataDirName = 'experiment_2018_01_19_21_48_47_0001'
#below is 250 iters training
#dataDirName = 'experiment_2018_01_21_20_01_42_0001'
#retrained with slightly different raise_vel 
#dataDirName = 'experiment_2018_02_05_06_48_23_0001'
#dataDirName = 'experiment_2018_02_05_08_48_04_0001'
#dataDirName = 'experiment_2018_02_05_08_48_04_0001'
#set all experiment variables (like policy directory) in tfuncs.buildExpDict




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
    

#######################################
#   OLD FILE NAMES
#######################################
##akanksha's policy
#dataDirName = 'AK_snapShots_pol_128_128_bl_MLP_blArch_32_32_batches_50000_bounded'
#dataDirName = 'exp-smplRwd_2018_07_11_20_07_10_0001'
#policy with random init states drawn from best performing states
#dataDirName = 'exp-000100110111_bl_16-16_pl_96-32_it_200_mxPth_200_nBtch_50000_2018_08_08_11_25_54_0001'
#dataDirName = 'exp-001001110111_bl_16-16_pl_64-64_it_200_mxPth_200_nBtch_200000_ALG_TRPO__2018_08_15_23_13_03_0001'
#dataDirName = 'exp-00001000011_bl_16-16_pl_64-64_it_200_mxPth_200_nBtch_50000_ALG_TRPO__2018_08_17_10_27_55_0001'
    
#attempts to reproduce GAE paper
#use this env : 
#envName = 'DartStandUp3d_GAE-v1'
#dataDirName = 'exp-100000000000_bl_128-64-32_pl_128-64-32_it_200_mxPth_200_nBtch_200000_ALG_TRPO__2018_08_20_09_17_31_0001'
#--v squirms on ground
#dataDirName = 'exp-100000000000_bl_64-64_pl_128-64-32_it_500_mxPth_1000_nBtch_200000_ALG_TRPO__2018_08_31_13_29_57_0001'
#Longest trained
#dataDirName='exp-100000000000_bl_64-64_pl_128-64-32_it_1000_mxPth_1000_nBtch_200000_ALG_TRPO__2018_09_03_09_59_00_0001'
#extended to 5000 iters - this is at ~3700 iters
#dataDirName='exp-100000000000_bl_64-64_pl_128-64-32_it_5000_mxPth_1000_nBtch_200000_ALG_TRPO__2018_09_04_12_22_25_0001'
#trained all 5k iters at one time, not retrained
#dataDirName='exp-100000000000_bl_64-64_pl_128-64-64_it_5000_mxPth_500_nBtch_200000_ALG_TRPO__2018_09_07_13_04_00_0001'
#old files at end of this file


    
#all old kima skeleton (with expanded abdomen and spine dof lims):
#dataDirname = 'exp-000000110110_bl_16-16_pl_96-32_it_200_mxPth_200_nBtch_50000_2018_07_22_17_31_46_0001'
#dataDirName = 'exp-000000110110_bl_16-16_pl_96-32_it_200_mxPth_200_nBtch_50000_2018_07_22_14_56_59_0001'
#dataDirName = 'exp-000000000110_bl_32-32_pol_96-64_iters_200_maxPath_2000_numBatch_50000_2018_07_21_09_03_26_0001'
#dataDirName = 'exp-000000110110_bl_32-32_pol_96-64_iters_200_maxPath_2000_numBatch_50000_2018_07_20_18_49_09_0001'
#using GAE paper getup reward
#dataDirName = 'exp-smplRwd_bl_32-32_pol_96-64_iters_150_maxPath_2000_numBatch_200000_2018_07_18_11_16_48_0001'

#simple getup COM height and action rwd, h x 100, var=.1
#dataDirName = 'exp-smplRwd_2018_07_18_07_32_34_0001'
#basic getup COM height and action rwd,com h x 1000, var=.01 action pen x 1
#dataDirName = 'exp-smplRwd_2018_07_17_23_18_32_0001'
#basic getup with only COM height and action rwd; com h x 10, action pen x 1
#dataDirName = 'exp-smplRwd_2018_07_17_13_34_26_0001'
#3rd of new exp reward formula, with 3rd weighting
#dataDirName = 'exp-smplRwd_2018_07_17_05_55_51_0001'
#2nd of new exp reward formula, with weighting on hand matching, feet stationary and com height above feet
#dataDirName = 'exp-smplRwd_2018_07_16_21_02_47_0001'
#1st of 3 with new exp formulation, using first set of weights - just falls back
#dataDirName = 'exp-smplRwd_2018_07_16_15_14_55_0001'
#no assist force, only get up  + min action reward
#dataDirName = 'exp-smplRwd_2018_07_14_05_06_17_0001'
#simplified getup with linear ht reward - tries harder to follow ball (?) but feet skip on ground
#dataDirName = 'exp-smplRwd_2018_07_11_20_07_10_0001'
#simplified getup with 100*ht reward - sticks feet well, doesn't straighten legs
#dataDirName = 'exp-smplRwd_2018_07_10_20_37_10_0001'
#simplified getup with 10*ht reward
#dataDirName = 'exp-smplRwd_2018_07_10_13_03_12_0001'
#simplified getup reward with raiseVel > 0 done condition
#dataDirName = 'exp-smplRwd_2018_07_09_18_54_00_0001'
#500 iters, getup pass 1 after groundtruth rl training verification
#dataDirName = 'experiment_2018_07_07_21_54_02_0001'

####################################
#Ground truth experiments
#250 iters, min action & max COM height
#dataDirName = 'exp-smplRwd_2018_07_07_15_32_14_0001'
#250 iters, min action and 1/10 and lift arm 9/10
#dataDirName = 'exp-smplRwd_2018_07_07_11_22_01_0001'
#250 iters, min action 1/3 + lift left arm 2/3
#dataDirName = 'exp-smplRwd_2018_07_06_19_26_33_0001'
#250 iters, min action + lift left arm - equal weight
#dataDirName = 'exp-smplRwd_2018_07_06_16_33_02_0001'
#200 iters train, minimize action
#dataDirName = 'exp-smplRwd_2018_07_06_13_50_47_0001'
#new skel, with modified lcp.cpp file that doesn't throw ODE assertion errors that cause rllab to hang
#dataDirName= 'exp-smplRwd_2018_07_06_10_02_32_0001'
#old skeleton- simple torque min with 32 iters
#dataDirName = 'exp-smplRwd_2018_07_06_07_56_04_0001'
#50 iters of simplified reward function old skel
#dataDirName = 'exp-smplRwd_2018_06_27_12_34_12_0001'
#250 iters, starts to get up - massive force
#dataDirName = 'experiment_2018_06_26_20_50_12_0001'
####################################################
#
# #akanksha's policy and environment
#envName = 'DartKimaStandUp-v1'
#dataDirName = 'AK_snapShots_pol_128_128_bl_MLP_blArch_32_32_batches_50000_bounded'

#kinematically moving ball target - bad
#dataDirName='experiment_2018_06_25_11_29_32_0001_KinMovBall'

#incomplte trianing
#dataDirName = 'experiment_2018_06_25_10_40_07_0001_incmpltTrain'
#below has short trajectories sucks
#dataDirName = 'experiment_2018_06_23_11_24_06_0001'


#fully trained with .5,1.0,0 force and different reward - flys past box, lifts feet : dataDirName = 'experiment_2018_06_26_04_48_09_0001'
#61 iters with .5,1.0,0 force - raises right leg high : dataDirName = 'experiment_2018_06_25_19_07_03_0001'

#policy trained with massive force - too much weight on COP dist, curls in a ball
#dataDirName = 'experiment_2018_06_25_15_40_37_0001'
#below has longer trajs
#dataDirName = 'experiment_2018_06_23_09_34_26_0001'