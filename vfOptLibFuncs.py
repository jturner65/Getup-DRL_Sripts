#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 06:30:52 2018

@author: john

functions originally in trpoLibFuncs used for value function optimization process

"""
import numpy as np
#for ordered dict
import collections


#class to hold functionality to build and consume a network used as a function approximator, with code to optimize input to maximize output
class vfOpt():
    def __init__(self, blRegressor, dart_env, numRandAssists=10, numMaxOptIters=100, minAlphaStep=.01):
        self.dart_env = dart_env
        self.numRandAssists = numRandAssists
        #max # of optimization iterations for grad descent
        self.numMaxOptIters = numMaxOptIters
        #min size alpha (lr) can be before we break out of adaptive ts loop
        self.minAlphaStep = minAlphaStep
        self.vfDataDict = self._buildVFNetFromBaseline(blRegressor)
        
    #build an MLP without the overhead of the baseline/regressor class,
    #passed existing baseline's regressor
    def _buildVFNetFromBaseline(self, blRegressor):
        #blRegressor =polDict['baseline']._regressor
        import lasagne
        import lasagne.layers as L
        import theano as T
        #import theano.tensor as T
        from rllab.core.network import MLP
        #architecture of baseline mean network
        blLayerShapes = blRegressor.get_param_shapes()
        #tuple to hold architecture
        tmpL = []
        for i in range(1,len(blLayerShapes)-2,2):
            tmpL.append(blLayerShapes[i][0])
        blArchTupl = tuple(tmpL)
        blNonlinearity = blRegressor._mean_network.layers[1].nonlinearity
        outNonLinearity = blRegressor._mean_network.output_layer.nonlinearity
        #print('Nonlinearity in blRegressor : {} | Output nonlinearity in blRegressor : {}'.format(blNonlinearity,outNonLinearity))
        #parameters of baseline mean network
        blParams = L.get_all_param_values(blRegressor._mean_network.output_layer)
    
        #build new network - make sure to match nonlinearity to source blregressor
        net = MLP(input_shape=(blLayerShapes[0][0],),
                output_dim=1,
                hidden_sizes=blArchTupl,
                hidden_nonlinearity=blNonlinearity,#lasagne.nonlinearities.rectify,
                output_nonlinearity=outNonLinearity,
                )
        #set net's parameters to be baseline mean network parameters
        L.set_all_param_values(net.output_layer,blParams)
        #use net's input variable
        X = net.input_layer.input_var
        #get net's output predictions
        pred = L.get_output(net.output_layer,deterministic=True)
        #build theano function mapping input to prediction (value function model)
        valueFunc = T.function([X],pred) 
        #build jacobian
        vfJacob = T.gradient.jacobian(pred[0], X)       #use consider_constant=<theano var> to set constant elements (?)
        
        #return net and valueFunc model    
        resDict = dict()
        resDict['X'] = X
        resDict['net'] = net
        resDict['valueFunc'] = valueFunc
        resDict['pred'] = pred
        resDict['vfJacob'] = vfJacob
    
        return resDict
    
    #provide ordered dictionary of best vf preds of assistance for passed q,qdot, ordered by score
    #vfDataDict : theano-derived net values and jacobians from baseline
    #    vfDataDict['X'] = X (input)
    #    vfDataDict['net'] = net (copy of baseline net)
    #    vfDataDict['valueFunc'] = valueFunc (functional representation)
    #    vfDataDict['pred'] = pred (output prediction layer)
    #    vfDataDict['vfJacob'] = vfJacob (value function jacobian w/respect to input X)
    #obs : observation from environment
    #returns res : result dictionary of up to numRandAssists scores/assistance proposals from obs(if any assist components were <0 in reward, they were discarded so may be <numRandAssists)
    def findVFOptAssistForObs(self, obs):
        #assistNegIDX is end idx of observation where assistance component begins
        obsPrfx, _, assistNegIDX = self.dart_env.getObsComponents(obs)
        print('findVFOptAssistForObs : obs len = {} assist neg idx = {}'.format(len(obsPrfx),assistNegIDX))
        return self._findVFOptAssistForState(obsPrfx, assistNegIDX)

    #find optimal assist for passed state
    #dart_env : reference to instanced dart env class
    #vfDataDict : value function dictionary
    #obsPrfx : current state without force component
    #assistNegIDX : - length of assist component of observation
    def _findVFOptAssistForState(self, obsPrfx, assistNegIDX):
        res = dict()
        dart_env = self.dart_env
        numRandAssists = self.numRandAssists
        #draw numRandAssists random force values, tossing anything that yields a non-positive result
        itrs = 0
        maxitrs = 50*numRandAssists  #to prevent infinite looping
        numVals = 0
        tmpPath = {}#dict(dart_env = dart_env)
        assistMultBnds = dart_env.getAssistBnds() # 2 x dim array of low and high assist components
        #setting min and max bounds of initial assist proposals in somewhat from actual assist bounds specified in env, due to convergence issues
        for i in range(len(assistMultBnds[0])):
            #low and high forces
            assistMultBnds[0][i]+=.00001
            assistMultBnds[1][i]-=.00001           
            
        X = self.vfDataDict['X']
        vf = self.vfDataDict['valueFunc']
        Jvf = self.vfDataDict['vfJacob']
        while (numVals < numRandAssists) and (itrs < maxitrs):
            itrs+=1
            #returns a random force mult, the commensurate force given the force mult bounds, and, lastly, the actual observation value (either the force or the force mult), depending on what the environment is set with.  
            # NOTE : very important that the environment setting matches the trained policy setting, or hilarity will ensue 
            #_, _, rndObsAssist = dart_env.getRandomAssist(assistMultBnds)  
            rndObsAssist = dart_env.getRandomAssist(assistMultBnds)  
            #build initial state observation dictionary in format expected
            tmpPath['observations']=[np.concatenate((obsPrfx,rndObsAssist))]
            val, optAssist = self._findOptAssist(tmpPath, X, vf, Jvf, assistNegIDX)

            #if val is none then bad/infeasible proposal for assist - shouldn't happen
            if (val is not None):            
            #if(val[0][0] > 0):
                #totStmt = ''.join(['Start assist {} optimized to {} yielded '.format(obsFrc,optAssist),'good vf approx : {:5.5f} '.format(val[0][0])])
                #keyed by rwrd, value is frc
                res[val[0][0]] = optAssist
                itrs = 0 #good value means reset iterations to get here
                numVals +=1
                #print (totStmt)
            else:
                totStmt = ''.join(['\t','Start assist {} optimized to {} claimed illegal by env :: value tossed '.format(rndObsAssist,optAssist)])
                print (totStmt)
            #print('init s:{} | opt force : {} | score : {:5.5f}'.format(s,optFrc, val[0][0]))
        #order results based on key value (score) 
        resDict = collections.OrderedDict(sorted(res.items(), reverse=True))            
        return resDict

   
    #set optimal assistive force given current state - called after reset, where force is originally set in rollout/env - overrides value specified in reset
    #overrides force and force mult, but leaves state of useSetForce flag alone
    def setOptAssistForState(self, obs):
        resDict = self.findVFOptAssistForObs(obs=obs)
        if(len(resDict) < self.numRandAssists):
            print('Incomplete Res Dict : res dict size : {} numRandAssists specified {} '.format(len(resDict), self.numRandAssists))        
        #set initial optimal force
        #score,optVal = list(resDict.items())[0]
        _,optVal = list(resDict.items())[0]
        #print('opt force for state : {:.5f}, {:.5f}'.format(optVal[0],optVal[1]))
        #set optimal force in env, and get new observation
        #this setting does not change state of useSetForce flag in env, since this is only consumed on reset.  We want next reset to persist with old behavior
        self.dart_env.setAssistForceDuringRollout(optVal, False)
        newObs = self.dart_env._get_obs()
        return newObs
    
    #set optimal assistance in environment given initial state - called before reset is called in env - sets necessary flags in env to use set assist specified here
    def setOptInitAssist(self, q, qdot):   
        obs = self.dart_env.dart_env.getObsFromState(q,qdot) 
        resDict = self.findVFOptAssistForObs(obs=obs)
        if(len(resDict) < self.numRandAssists):
            print('Incomplete Res Dict : res dict size : {} numRandAssists specified {} '.format(len(resDict), self.numRandAssists))        
        #set initial optimal force
        #score,optForce = list(resDict.items())[0]
        _,optForce = list(resDict.items())[0]
        #print('opt force for state : {}'.format(optForce))
        #set optimal force in env, and get new observation
        #NOTE : need to use setFrcMultFromFrc to set this so flags in env are set appropriately to use it and not a random force
        self.dart_env.setFrcMultFromFrc(optForce)
        newObs = self.dart_env._get_obs()
        return newObs
    
    def setNumRandAssists(self, num):
        self.numRandAssists = num
        
    
    #given state and jacobian of value function w/respect to state, find optimal force value
    #tmpPath : single observation of state, including assist proposal
    #X : input representation of net
    #vf : value function functional rep
    #vfJacob : jacobian of value function
    #assistNegIDX : negative index of assist component of observation (i.e. where assist component begins))
    def _findOptAssist(self,tmpPath, X, vf, vfJacob, assistNegIDX):
        #copy of state/observation to use to optimize force components - format necessary for structure of opt process
        stateToOpt = list([np.array([x for x in tmpPath['observations'][0]])])
        dart_env = self.dart_env
        #print ('--{}--'.format(stateToOpt))
        oldScore = -100000000
        oldBestScore = oldScore
        vfEval = vf(stateToOpt)
        i = 0
        numMaxOptIters = self.numMaxOptIters
        minStep = self.minAlphaStep
        #prevent infinite execution - set to 100 as default
        while (i<numMaxOptIters):
            i+= 1
            #evaluate jacobian passed 
            jacVal = vfJacob.eval({X:stateToOpt})            
            assistJacVal = jacVal[0][0][assistNegIDX:]
            #print("gradient : delX={:5.5f} delY={:5.5f} delZ={:5.5f}".format(jacVal[0][0][-3],jacVal[0][0][-2],jacVal[0][0][-1]))            
            alpha = 1.0    
            badAssistVal = False
            #adaptive step size - don't get smaller than minStep (default set to be .01)
            while (alpha > minStep) :
                modVal = alpha * assistJacVal
                #reapply until gets worse
                while True:
                    #copy last assist values
                    #lastVals = stateToOpt[0][-2:]
                    #mod values
                    stateToOpt[0][assistNegIDX:] += modVal
                    #copy old score as old best score
                    oldBestScore = oldScore
                    #copy previous iteration's score as old score
                    oldScore = vfEval[0][0]
                    #evaluate new state
                    vfEval = vf(stateToOpt)
                    #if new evaluation is at least as good as old score, or either value is negative, break out of loop
                    validAssist = dart_env.isValidAssist(stateToOpt[0][assistNegIDX:])
                    #print('\t\tstate {:5.5f},{:5.5f},{:5.5f} with alpha:{} -> score:{} | oldScore:{} | new is valid :  {}'.format(stateToOpt[0][-3],stateToOpt[0][-2],stateToOpt[0][-1],alpha,vfEval[0][0],oldScore,validAssist))
                    #TODO Shouldn't this be vfEval[0][0] ?
                    #if (vfEval[0] <= oldScore) or not validFrc :#(stateToOpt[0][-3] < 0) or (stateToOpt[0][-2] < 0):
                    if (vfEval[0][0] <= oldScore) or not validAssist :#(stateToOpt[0][-3] < 0) or (stateToOpt[0][-2] < 0):
                        if not validAssist : #(stateToOpt[0][-3] < 0) or (stateToOpt[0][-2] < 0):
                            badAssistVal = True
                        #print('\t\t\talpha too big :{} '.format(alpha))
                        break
                    
                #got worse so halve alpha and restore state and score
                alpha *= .5
                stateToOpt[0][assistNegIDX:] -= modVal #undo mod
                vfEval = vf(stateToOpt)#vfEval[0] = oldScore
                oldScore = oldBestScore
                if(badAssistVal):
                    if(alpha > minStep):
                        badAssistVal = False
                        #TODO add handling of negative assist values by end of opt process?
                
                #print("\tBest force for alpha {}} | alpha : {:4.4f} | old score {} : score {}".format(stateToOpt[0][-2],stateToOpt[0][-1], alpha,oldScore, vfEval[0]))            
            #print("Best state for gradient : delX={:5.5f} delY={:5.5f} -> {:5.5f},{:5.5f} | alpha : {:4.4f} | old score {} : score {}".format(jacVal[0][0][-2],jacVal[0][0][-1],stateToOpt[0][-2],stateToOpt[0][-1], alpha,oldScore, vfEval[0][0]))            
            #print("")
        #calcVfEval  = vfEval[0][0]
        #vfEval = vf(stateToOpt)
        #print("Best state :  {:5.5f},{:5.5f} | last score {} | calc score {}".format(stateToOpt[0][-2],stateToOpt[0][-1], vfEval[0][0], vfEval[0][0]))
        # stateToOpt idx 0 is final state in observations            
        optAssist = stateToOpt[0][assistNegIDX:]
        # if(stateToOpt[0][-1] < 0) or (stateToOpt[0][-2] < 0):
        #     return stateToOpt, [[-1]], optAssist
        # return stateToOpt, vfEval, optAssist
        #z force can be negative (idx -1) but x and y shouldn't
        if not dart_env.isValidAssist(optAssist) : 
            return None, optAssist
        return vfEval, optAssist   
    
