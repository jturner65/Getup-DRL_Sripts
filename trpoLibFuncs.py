# -*- coding: utf-8 -*-
import os, joblib, ast
#trpo 
import rllab.misc.logger as logger
#from rllab.misc.ext import set_seed
#from rllab.algos.trpo import TRPO
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
#plot results
import matplotlib.pyplot as plt

import numpy as np

#default dictionary
from collections import defaultdict



#return list of all subdirs within passed directory
def getSubdirList(expDict):
    baseDir = expDict['envName'] +'_res'
    return [name for name in os.listdir(baseDir) if (os.path.isdir(os.path.join(baseDir, name))) ]#and ('snapShots_' in name)]


#set reward functions based on directory name - MUST BE CORRECT FORMAT (exp-<xxx> where xxx is binary digits corresponding to reward components) OR EVERYONE DIES! HORRIBLY!!
def setEnvRwdFuncs(dataDirName):
    #to be able to modify global value for using simple reward
    import gym.envs.dart.dart_env_2bot as dartEnv2bot
    #assumes configuration is using appropriate format that encodes rewards as binary digit string in dir name
    binDigStr = dataDirName.split('_')[0].split('-')[1]
    rwdList = dartEnv2bot.DartEnv2Bot.setGlblRwrdsFromBinStr(binDigStr=binDigStr)
    #unlikely that any training being done in this environment - set to false so that env instances a viewer
    verifyTrain = dartEnv2bot.DartEnv2Bot.setTrainState(False)
    print('setEnvRwdFuncs : Training is set to : {}'.format(verifyTrain))

    return rwdList


#build all dicts and put in single dictionary - this is only function that should be called externally
def buildAllExpDicts(dataDirName=None, envName='DartStandUp3d_2Bot-v1', isNormalized=False, initStatesCSVName='testStateFile.csv',resumeTrain=True, useNewEnv=False, launchEnv=True):
    expDict = _buildExpDict(dataDirName=dataDirName, envName=envName, isNormalized=True, initStatesCSVName=initStatesCSVName, resumeTrain=True)
    polDict, trainDict, assistDict = _buildExperiment(expDict, useNewEnv=True)
    dataDict = {}
    dataDict['expDict']=expDict
    dataDict['polDict']=polDict
    dataDict['trainDict']=trainDict
    dataDict['assistDict']=assistDict
    dataDict['dartEnv'] = polDict['dartEnv']
    return dataDict
    

#build experiment dictionary
#if dataDirName != none then experimental data(policy,baseline, etc) 
#   will be loaded from dataDirName and expdict settings will be overwritten
#
def _buildExpDict(dataDirName=None, envName='DartStandUp3d_2Bot-v1', isNormalized=False, initStatesCSVName='testStateFile.csv',resumeTrain=True):
    #Loaded trained policy/baseline will override these settings
    #variables defining experiment
    expDict = dict()
    
    dataDirNameExists = dataDirName is not None

    #set dart_env_2bot reward vals based on string in dataDirName
    # setGlblRwrdsFromBinStr    
    #values below are overridden by loaded data if using pretrained policy    
    # number of iterations to train or train further (if loading existing policy)
    expDict['numIters'] = 100# overwritten by saved policy if neceswsary
    #whether to resume training or start from scratch - start from scratch always if dataDirName is none
    expDict['resumeTrain'] = resumeTrain and dataDirNameExists
    #baseline type : currently either 'MLP' or 'linear' - linear ignores blMlpArch
    expDict['blType'] = 'MLP'
    #if baseline is mlp, use this as mean architecture - uses same arch for std unless otherwise specified
    #32,32 baseline consistently fails
    expDict['blMlpArch']=(16,16)#(32,32)#(16,)#(8,8)# overwritten by saved policy if neceswsary
    #environment name
    expDict['envName'] = envName
        
    expDict['isGymEnv'] = True
    #policy architecture - # of perceps per hid lyr
    expDict['polNetArch'] = (64,64)# overwritten by saved policy if neceswsary
    # number of batches
    expDict['numBatches'] = 50000#50000 #both are built# overwritten by saved policy if neceswsary
    #for testing, use stochastic or deterministic (mean) policy stochastic
    expDict['useRndPol'] = False
    #list of directories within envName base dir - for testing
    expDict['subDirList'] = getSubdirList(expDict)
    
    #use generated, or override generated with given, file name/file directory - must be subordinate to <envName>_res dir
    expDict['useGenDirName'] = not(dataDirNameExists) #Note if a new experiment is run, the results in this directory will be overwritten
    #if not using generated dir name (useGenDirName set false above), this is directory name of location to be loaded
    #use idx 1 or 3
    expDict['ovrd_dataDirName'] = dataDirName
    #name of csv file holding pre-derived initial env states
    expDict['savedInitStates'] = initStatesCSVName
    #use normalized environment?
    expDict['isNormalized'] = isNormalized
    expDict['isNormalized_pref'] = isNormalized
    #based on name or specification - if we want to force an env to be non-/normalized we use isNormalized_pref
    expDict['isNormalized_byName'] = isNormalized
    #attempt to overwrite these defaults with info from dataDirName
    if dataDirNameExists :    
        import ast
        try :   #if fails dataDirName probably is incorrect format
            expNameDict, _, _ = expDirNameToString(dataDirName)
            #expNameDict['rwds']
            #bl arch is idx 2; pol archi is idx 4; 
            expDict['blMlpArch'] = ast.literal_eval(expNameDict['blArch'])
            expDict['polNetArch'] = ast.literal_eval(expNameDict['polArch'])
            expDict['numIters'] = int(expNameDict['numIters'])
            expDict['numBatches'] = int(expNameDict['batchSize'])   
            expDict['isNormalized_byName'] = expNameDict['normedEnv']        
        except:
            pass
        
    #dictionary holding all training runs
    expDict['trainingInfoDict'] = loadTrainInfo(expDict)

    print('Loaded ExpDict : ')
    for k,v in expDict.items():
        if 'trainingInfoDict' not in k and 'subDirList' not in k:
            print('{} : {}'.format(k,v))
    return expDict

#either build a new environment, policy and algorithm or load an existing one
#snapShotDir is pre-made directory to put video snapshot
def _buildExperiment(expDict, useNewEnv=False, launchEnv=True):
    polDict = {}
    #if saving video and polDict already made
        #want logger initially set off so environment not saving vids
    logger.set_snapshot_dir(None)
    logger.set_snapshot_mode('all')
       
    if(expDict['resumeTrain']):
        fileDir, data = loadPolicyEnv(expDict)
    #attempt to resume training resume training of saved algorithm, using saved pkl file
    if (not expDict['resumeTrain']) or (data == None) : 
        #if saved data not found build new policy and algo
        print('!!!Initializing new policy and algo for this experiment')
        expDict['isNormalized'] = expDict['isNormalized_pref'] 
        env = loadEnv(expDict)
        #directory to save results for this run - built off policy architecture
        baseDir, fileSaveDir = makePklDirNames(expDict)
        makeDir(baseDir)
        polDict['snapShotDir'] = makeDir(fileSaveDir)        
        polDict['policy'] = GaussianMLPPolicy(
            env_spec=env.spec,
            #must be tuple - if only 1 value should be followed by comma i.e. (8,)
            hidden_sizes=expDict['polNetArch']
        )
        polDict['baseline'] = defBaseLine(env, expDict)        
        startIter = 0
    else :
        #data found and wanting to resume training
        print('!!!Existing policy found at '+fileDir)
        
        env = data['env']
        #add ref to dart env class to more easily call it
        try : #if env is normalized then it is wrapped
            gymEnv = env.wrapped_env
        except : #if this triggers then we're not using a normalized environment
            gymEnv = env
        
        if ((gymEnv.env_id not in expDict['envName']) or useNewEnv) :
            gymEnv.env.unwrapped.close()
            print('\nNOTE : Loaded Policy trained on environment {} which is different than specified.  Loading environment specified in expdict : {}!!!'.format(env.wrapped_env.env_id , expDict['envName']))
            expDict['isNormalized'] = expDict['isNormalized_pref'] 
            env = loadEnv(expDict)
        logger.set_snapshot_dir(None)
        logger.set_snapshot_mode('all')
        #env.render()
        
        polDict['snapShotDir'] = fileDir
        polDict['policy'] = data['policy']
        polDict['baseline'] = data['baseline']
        polDict['data'] = data
        startIter = data['itr']+1
        updateExpDictForLoadedData(expDict, data)
        print('Data loaded for policy trained over '+str(startIter)+' iterations')
        #polDict['data'] = None
    
    #close env if not wanting to launch it
    if not launchEnv :
        gymEnv.env.unwrapped.close()        
    
    #add # of already trained iterations
    expDict['trainedIters'] = startIter
    trainDict = buildTrainDict(startIter, expDict['numIters'], expDict)
    #add ref to base env to polDict so we don't have to carry it around everywhere
    #NOTE env != dart_env.  env is rllab base environment, dart_env is instanced environment
    polDict=setPolDictEnv(polDict, env)#don't need to return polDict since passed by ref, but just to make it obvious that polDict changes in function
    
    #dictionary describing assistance values for manually setting them,e ither force based or other
    if polDict['dartEnv'].assistIsFrcBased : 
        assistDict = defaultdict(int, {'chgUseSetFrc':1, 'useSetFrc':1,'passedMultNotFrc':polDict['dartEnv'].useMultNotForce})
    else :
        #TODO determine assist flags if not using force assist 
        assistDict = defaultdict(int)  
    #disable viewer
    #polDict['dartEnv'].setViewerDisabled(True)
    #whether we use assistive force or assistive force multiplier to train/consume policy/baseline
    #SET THIS FROM VALUE IN ENVIRONMENT
    try:
        #old envs didn't use this
        expDict['bl_useMultNotForce'] = polDict['dartEnv'].useMultNotForce
    except:
        expDict['bl_useMultNotForce'] = False

    return polDict, trainDict, assistDict

#add an environment to existing polDict dictionary
def setPolDictEnv(polDict, env):
    polDict['env'] = env
    #add ref to dart env class to more easily call it
    try : #if env is normalized then it is wrapped
        polDict['dartEnv'] = env.wrapped_env.env.unwrapped
    except : #if this triggers then we're not using a normalized environment
        polDict['dartEnv'] = env.env.unwrapped
    return polDict

#load policy build policy dict without env
def loadNewPolDict(expDict):
    fileDir, data = loadPolicyEnv(expDict)
    env = data['env']
    #close environment if not none
    if(env is not None):
        try : #if env is normalized then it is wrapped
            gymEnv = env.wrapped_env
        except : #if this triggers then we're not using a normalized environment
            gymEnv = env

        gymEnv.env.unwrapped.close()
        
    logger.set_snapshot_dir(None)
    logger.set_snapshot_mode('all')
    polDict = {}
    polDict['snapShotDir'] = fileDir
    polDict['policy'] = data['policy']
    polDict['baseline'] = data['baseline']
    updateExpDictForLoadedData(expDict, data)   
    return polDict

def defBaseLine(env, expDict):
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
    
    if (expDict['blType'] == 'linear'):
        return LinearFeatureBaseline(env_spec=env.spec)
    elif (expDict['blType'] == 'MLP'):
        #use regressor_args as dict to define regressor arguments like layers 
        regArgs = dict()
        regArgs['hidden_sizes'] = expDict['blMlpArch']
        regArgs['std_hidden_sizes']= expDict['blMlpArch']
        #normalize inputs and outputs?
        regArgs['normalize_inputs'] = False
        regArgs['normalize_outputs'] = False
        bl = GaussianMLPBaseline(env_spec=env.spec, regressor_args=regArgs)
        return bl
    else:
        print('unknown baseline type : ' + expDict['blType'])
        return None
          
#load environment
def loadEnv(expDict, record=False):
    if(expDict['isNormalized']):
        print('Building Normalized Environment for {}'.format(expDict['envName']))
        if(expDict['isGymEnv']):
            env = normalize(GymEnv(expDict['envName'],record_video=record, record_log=record))
        else :
            env = normalize(expDict['envName'])
    else :#if not normalized, needs to be gym environment
        print('Building Non-Normalized Environment for {}'.format(expDict['envName']))
        env = GymEnv(expDict['envName'],record_video=record, record_log=record)
    return env

#modify expDict to account for config of loaded data - 
#check if linear baseline
def updateExpDictForLoadedData(expDict, data):
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

#    expDict['blType'] = #'MLP'
    #if baseline is mlp, use this as mean architecture - uses same arch for std unless otherwise specified
    #if baseline is linear, this is ignored
    oldBLType = expDict['blType']
    oldBLArch = expDict['blMlpArch']
    oldPolArch = expDict['polNetArch']
    if(LinearFeatureBaseline == type(data['baseline'])):
        expDict['blType'] = 'linear'
    else :
        expDict['blType'] = 'MLP'
        expDict['blMlpArch']=data['baseline']._Serializable__args[3]['hidden_sizes']
    #policy architecture - # of perceps per hid lyr
    expDict['polNetArch'] = data['policy']._Serializable__args[1]
    print('Updated expDict with data from loaded policy/baseline')
    print('BL Type specified : {} | BL Type found (and used) : {} '.format(oldBLType,expDict['blType']))
    print('BL arch changed from {} to {}'.format(oldBLArch,expDict['blMlpArch']))
    print('Policy arch changed from {} to {}'.format(oldPolArch,expDict['polNetArch']))
    
#build experiment, train and make recording
def makeRecording(expDict):
    from rllab.algos.trpo import TRPO

    polDict = {}
    #if saving video and polDict already made
    _, snapShotDir = makePklDirNames(expDict)
    #set before env made so that vids are saved
    logger.set_snapshot_dir(snapShotDir)
    #probably why crashes when writing vids - tries to overwrite file that isn't done being saved
    logger.set_snapshot_mode('last')
       
    fileDir, data = loadPolicyEnv(expDict)
    #want to load environment here because we want to set up to enable videos
    print('!!!Initializing new environment for this experiment')
    #use name-dictated, or secondarily preference dictated, values for normalization
    expDict['isNormalized'] = expDict['isNormalized_byName'] 
    env = loadEnv(expDict, record=True)
    print(fileDir)
    polDict['snapShotDir'] = fileDir
    polDict['policy'] = data['policy']
    polDict['baseline'] = data['baseline']
    startIter = data['itr']+1
    print('Data loaded for policy trained over '+str(startIter)+' iterations')
    #only 1 iteration
    trainDict = buildTrainDict(startIter, 1, expDict)
    #setupRunExp(polDict, trainDict, env, seed=None, n_parallel=1, plot=False)
    algo = TRPO(
        env=env,
        policy=polDict['policy'],
        baseline=polDict['baseline'],
        batch_size=trainDict['batch_size'],
        whole_paths=trainDict['whole_paths'],
        max_path_length=trainDict['max_path_length'],
        n_itr=trainDict['n_itr'],
        discount=trainDict['discount'],
        step_size=trainDict['step_size'],
        start_itr=trainDict['start_itr']
    )
    algo.train()
    return env, polDict, trainDict
    
#return a display-friendly name describing experiment in terms of baseline
#type and architecture, and policy arch
def getExpNameDisp(expDict):
    res = 'MLP Policy with {} layers and '.format(expDict['polNetArch'])
    if(expDict['blType'] == 'MLP'):
        res = res + 'MLP bl with {} arch.'.format(expDict['blMlpArch']) 
    elif(expDict['blType']=='linear'):
        res = res + 'linear baseline' 
    else :
        res = res + 'unknown baseline'
    return res

#will load csv holding training data info into a dictionary of lists
#will also build polynomials for each column based on # of epochs and samples per batch
def loadTrainInfo(expDict):
    import csv
    _, fileDir = makePklDirNames(expDict)
    fName = os.path.join(fileDir,'progress.csv')
    try:        
        rdr = csv.reader(open(fName, 'r'))
        res = defaultdict(list)
        colNames = next(rdr)
        numCols = len(colNames)
        for row in rdr:
            for k in range(numCols):
                res[colNames[k]].append(float(row[k].strip()))
        print('Existing Training Info loaded from location : %s' % (fName))
    except (FileNotFoundError, RuntimeError, TypeError, NameError):
        print('!!!!Training Info not found for envName : %s and policy arch tuple : %s in file name %s - no training plots will be possible.' %(expDict['envName'], str(expDict['polNetArch']), fName))
        return None
    
    #correct # of iters to be # of rows written
    expDict['numIters']=len(res[colNames[0]])
    #each data type has a data point per iter : expDict['numIters'];
    #expDict['numBatches'] is # of samples per iter
    # to normalize per sample build eq of data points per iter * numBatches
    numSamplesPerEpoch = expDict['numBatches'] 
    print('Num Samples Per Epoch : {}'.format(numSamplesPerEpoch))
    numEpochs = expDict['numIters']
    xVec = np.linspace(0,(numEpochs-1), num=numEpochs)
    xVec *= numSamplesPerEpoch
    #maxXVec
    #build eqs
    newRes = defaultdict(list)
    newRes['xVec'] = list(xVec)
    #scale to be between -1 and 1 for legendre polynomial
    normXVec = 2*xVec/xVec[-1] - 1
    for k,v in res.items():
        newRes[k]=v
        newK = '{}_coeffs'.format(k)
        #print('k:{} | normXVec len : {} | v len : {}'.format(k,len(normXVec), len(v)))
        newV = np.polynomial.legendre.legfit(normXVec, v, deg=50)
        #newV = np.polyfit(xVec, v, deg=10)
        newRes[newK]=newV
    res = newRes
    
    return res
    
#will load trained policy and environment held in pkl named fileName
def loadPolicyEnv(expDict):
    _, fileDir = makePklDirNames(expDict)
    pkl_fileName = os.path.join(fileDir,'params.pkl')
    try:
        data = joblib.load(pkl_fileName)
        print('Existing Data loaded from location : %s' % (pkl_fileName))
    except (FileNotFoundError, RuntimeError, TypeError, NameError):
        print('!!!!Data not found for envName : %s and policy arch tuple : %s in file name %s' %(expDict['envName'], str(expDict['polNetArch']), pkl_fileName))
        data = None
        pass

    #data is dictionary holding algo, iter, env, baseline, policypolValTpl, envName, numIter=1000, numBatch=4000, resumeTrain=False, isGymEnv=True, snapShotDir = None
    return fileDir, data


#make dir if not exists
def makeDir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    return dirName

#build file name based on policy arch tuple and name of environment
#recognize overrides here if any
def makePklDirNames(expDict):
    baseDir = expDict['envName'] +'_res'
    if(True == expDict['useGenDirName']):#use generated name for subdirectory
        tmpRes = 'snapShots_pol_' + '_'.join([str(x) for x in expDict['polNetArch']]) + '_bl_'+expDict['blType']
        if ('MLP' in expDict['blType'] ):    
            tmpRes = tmpRes + '_blArch_'+ '_'.join([str(x) for x in expDict['blMlpArch']])
        tmpRes = tmpRes + '_batches_'+str(expDict['numBatches'])
    else :
        tmpRes = expDict['ovrd_dataDirName']
    fileName = os.path.join(baseDir,tmpRes)
    return baseDir, fileName

#build training dictionary - values to use to train TRPO algorithm
#Can be run multiple times if values change
def buildTrainDict(start_itr, numIter, expDict):
    trainDict={}
    trainDict['n_itr'] = (numIter + start_itr)
    trainDict['start_itr'] = start_itr
    #default values - batch_size is # of individual samples per iteration
    #this will be met or exceeded if a trajectory has not ended by the time this many samples has been pulled
    trainDict['batch_size'] = expDict['numBatches']
    trainDict['max_path_length'] = 1000
    trainDict['whole_paths'] = True
    trainDict['discount'] = 0.99
    trainDict['step_size'] = 0.01
    
    return trainDict
    

#train algorithm for TRPO using passed policy and environment
def trainPolicy(polDict, trainDict):
    from rllab.algos.trpo import TRPO
    env = polDict['env']
        #set logger to save last result to snapShotDir
    logger.set_snapshot_dir(polDict['snapShotDir'])
    logger.set_snapshot_mode('last')
    #setupRunExp(polDict, trainDict, env, seed=1, n_parallel=1, plot=False)
    
    algo = TRPO(
        env=env,
        policy=polDict['policy'],
        baseline=polDict['baseline'],
        batch_size=trainDict['batch_size'],
        whole_paths=trainDict['whole_paths'],
        max_path_length=trainDict['max_path_length'],
        n_itr=trainDict['n_itr'],
        discount=trainDict['discount'],
        step_size=trainDict['step_size'],
        start_itr=trainDict['start_itr']
    )
    algo.train()
    
    return algo

def getResResultsFilename(polDict, expDict, iters, frcDim=1, testType=0):
    resName = polDict['snapShotDir']+os.sep+expDict['blType']
    if(expDict['blType'] == 'MLP'):
        resName = resName +'_blMlpArch_' + str(expDict['blMlpArch'][0]) + '_'+str(expDict['blMlpArch'][1]) 
    else:
        resName = resName + '_linear'
    #for force type 2 - save batch size and other info re:test method
    if(2 == frcDim):
        resName = resName + '_btchSz_'+str(expDict['numBatches'])
        resName = resName +'_testTyp_'+str(testType)

    resName = resName + '_testRuns_'+str(iters)+'_testRest.txt'
    return resName


###########################
########for optimization tests
#draw numStates random states from environment, and save them to filename
#poseDel == amount of variation in uniform sample of perturbation
#dart_env == env.wrapped_env.env.unwrapped
def saveNumRndStatesForSkel(dart_env, filename, numStates=50, posDel=None):
    stateRes = []
    for i in range(numStates):
        qpos, qvel = dart_env.getRandomInitState(posDel)
        #copy qpos and then qvel into single list
        tmpRes = [x for x in qpos]
        for x in qvel :
            tmpRes.append(x) 
        stateRes.append(tmpRes)
    #write list out to file as csv
    f = open(filename, 'w')
    for state in stateRes:
        resStr='{}'.format(state)
        f.write('{}\n'.format(resStr[1:-1]))#remove beginning and ending brackets from string rep of list
    f.close()
    return stateRes
       
#load pregenerated states to use them to set initial state for skeleton
#returns list of q and list of qdot
def loadRndStatesForSkel(dart_env, filename):
    f = open(filename, 'r')
    src_lines = f.readlines()
    f.close()
    qposList = []
    qvelList = []
    #get skel state and skel state-dot to determine length - each line in file should be 2x length
    qpos, qvel = dart_env.getRandomInitState()
    qSize = len(qpos)
    for line in src_lines:
        lineParse = line.split(',')
        tmpList = [float(n.strip()) for n in lineParse]
        qposList.append(tmpList[:qSize])
        qvelList.append(tmpList[qSize:])
        
    return qposList, qvelList


#load particular set of initial states and state dots
#returns must always be in list of lists of q and list of lists of qdots
#idxsToUse : list of idxs to use, None means use all
def loadInitStates(dart_env,expDict, idxsToUse=None):
    #initial state and state dot for CMA optimization
    if ('savedInitStates' not in expDict) or (expDict['savedInitStates'] is None):
        print('unable to find saved initial states - no file name specified in expDict')
        return [], []
    initQList, initQdotList = loadRndStatesForSkel(dart_env,expDict['savedInitStates'])
    #find optimal value for idx 0 of pre-calced init q and qdot
    if(idxsToUse is not None):
        initQList = [initQList[x] for x in idxsToUse]
        initQdotList = [initQdotList[x] for x in idxsToUse]
    return initQList, initQdotList

###########################
    
    
#convert a string of a list of strings of numerics to a numpy array
def convStrToNPAraFloat(listStr):
    tmpList = ast.literal_eval(listStr)
    print(tmpList)
    tmpList = [float(n.strip()) for n in tmpList]
    return np.asarray(tmpList, dtype=np.float64)

#convert a string of a list of lists to a 2d numpy array
def convStrTo2dNPAraFloat(listStr):
    tmpList = ast.literal_eval(listStr)
    #if type(tmpList) is list: #single list of elements
    tmpList = [x for x in tmpList]
    #else : #is tuple        
    #    tmpList = [[x] for x in tmpList]
    return np.asarray(tmpList, dtype=np.float64)
    
#listStr =  line.split(':')[-1].strip()   
#read srcFile into list of strings
#test results are a list of 2-tuples
#a float value representing force multiplier
#a dictionary of results, keyed by strings and values == lists of vals
#needs to be put in format : list of tuples with each tuple :
#idx 0 is frcmult (force applied is frcmult * mg @ pi/4 +x, +y)
#idx 1 is dictionary of results with keys 'advantages' and 'rewards'
#each value is a list of advantages and rewards at timesteps t-idx
#read until src_lines[i] == '______\n'
def readTestResults(filename):  
    datList = []
    f = open(filename, 'r')
    src_lines = f.readlines()
    f.close()
    tmpDataDict = {}
    dataTypeList = ['advantages_raw','advantages','rewards','returns']
    data2DtypeList = ['actions','observations']
    dataExpDict = {}
    for line in src_lines:
        #frc has prefix frc
        if 'frc' in line :
            frcLine = line.split(':')[-1].strip()
            if(',' in frcLine):#2d force vector
                frcStrList = frcLine.split(',')
                frc = (float(frcStrList[0].strip()),float(frcStrList[1].strip()))
            else :
                frc = float(frcLine)
#            if(1 == numFrcDofs):#force magnitude at pi/4
#                frc = float(line.split(':')[-1].strip())
#            elif(2 == numFrcDofs):#x/y values of force.  in file as 'frc:x,y'
#                frcStrList = line.split(':')[-1].split(',')
#                frc = (float(frcStrList[0].strip()),float(frcStrList[1].strip()))
        elif 'expDict' in line:
            kvList = line.split(':')[-1].split('|')
            dataExpDict[kvList[0].strip()]=kvList[1].strip()            
        elif '______' in line :      #demarcation between iterations of tests
            tup = (frc, tmpDataDict)
            datList.append(tup)
            frc = -1
            tmpDataDict = {}
        else:
            lineType = line.split(':')[0].strip()
            if lineType in dataTypeList:
                tmpDataDict[lineType] = convStrToNPAraFloat(line.split(':')[-1].strip())
            elif lineType in data2DtypeList:
                tmpDataDict[lineType] = convStrTo2dNPAraFloat(line.split(':')[-1].strip())
    return datList


#write datList into fName file, prefixed by expdict, which describes experiment-relevant info
def writeTestResults(fName, datList, expDict, frcDim):
    f = open(fName, 'w')
    for k,v in expDict.items() : 
        f.write('expDict:{} | {}\n'.format(k,v))
    f.write('\ndata\n\n')
    for item in datList:
        #tuple from list
        #a float value representing force multiplier
        #a dictionary of results, keyed by strings and values == lists of vals
        frc = item[0]#either a float value or a tuple
        path = item[1]#{}'.format(['{:.3f}'.format(i) for i in self.frcOffset]))
        if(1 == frcDim):#force magnitude at pi/4
            f.write('frc:{:.5f}\n'.format(frc))
        elif(2 == frcDim):#x/y values of force as tuple
            f.write('frc:{:.5f},{:.5f}\n'.format(frc[0],frc[1]))            
        f.write('advantages : {}\n'.format(['{:.5f}'.format(i) for i in path['advantages']]))#['{:.3f}'.format(i) for i in v]))
        f.write('advantages_raw : {}\n'.format(['{:.5f}'.format(i) for i in path['advantages_raw']]))#['{:.3f}'.format(i) for i in v]))
        f.write('observations:{}\n'.format('['+ '],['.join(','.join('{:.5f}'.format(i) for i in x) for x in path['observations'])+']'))
        f.write('actions:{}\n'.format('['+ '],['.join(','.join('{:.5f}'.format(i) for i in x) for x in path['actions'])+']'))#['{:.5f}'.format(i) for x in path['actions'] for i in x]))
        f.write('rewards : {}\n'.format(['{:.5f}'.format(i) for i in path['rewards']]))#.format(k,v))#['{:.3f}'.format(i) for i in v]))
        f.write('returns : {}\n'.format(['{:.5f}'.format(i) for i in path['returns']]))#.format(k,v))#['{:.3f}'.format(i) for i in v]))
        f.write('______\n')
    f.close()
          

#retrain baseline with explicit examples
#TODO specify specific initial states for retrianing rollouts?
def retrainBaseline(dataDict, rtrnDict, assistDict):    
    polDict = dataDict['polDict']

    dartEnv = polDict['dartEnv']
    usePolicyArgsDict = defaultdict(int,{'rndPolicy':rtrnDict['rndPolicy'], 'renderRes':rtrnDict['renderRes']})
    #usePolicyArgsDict = {'rndPolicy':rtrnDict['rndPolicy'], 'renderRes':rtrnDict['renderRes'], 'recording':False, 'debug':False, 'optOnlyBad':True}    
    assistDict['passedMultNotFrc']=0
    paths = []
    path_bls =[]
    i=0
    rejStr = 'rejBad' if rtrnDict['rejectBad'] else 'rejGood'
    #bounds from which to generate random values to use in optimization
    #can also use dartEnv.self.frcBnds
    assistMultBnds = dartEnv.getAssistBnds()
    #perform rollouts with specific assist values
    while i < rtrnDict['runIters']:
    #for i in range(iters):
        if(rtrnDict['setFrc']):
            rndAssist = dartEnv.getRandomAssist(assistMultBnds)
            dartEnv.setAssistDuringRollout(rndAssist, assistDict)
            #dartEnv.setForceMult(np.random.uniform(low=0.4, high=0.5),np.random.uniform(low=0.0, high=0.1))
        path, path_bl,_ = usePolicy(dataDict, optsDict=usePolicyArgsDict)#rtrnDict['rndPolicy'], renderRes=rtrnDict['renderRes'])
        #save path if either :
        #we're rejecting the bad ones and the path is longer than what is considered bad (we use  no bad rollouts and all good rollouts) OR
        #we're not rejecting bad ones, the path is 'bad' or rnd draw is less than "keep" rate setting (we use all bad and keepRate% of good rollouts)
        if (rtrnDict['rejectBad'] and (len(path['returns']) > rtrnDict['badPathLen'])) or \
            (not rtrnDict['rejectBad'] and ((len(path['returns']) <= rtrnDict['badPathLen'] ) or \
                                             (np.random.uniform(low=0.0, high=1.0) <= rtrnDict['keepRate']))): 
            print("{} iter {}".format(rejStr,i))
            i+=1
            paths.append(path)
            path_bls.append(path_bl)
    #fit baseline to new paths
    for i in range(rtrnDict['retrainIters']):
        polDict['baseline'].fit(paths)
    return polDict['baseline'], paths, path_bls

#retrain baseline with explicit examples using displacement as assistance
def retrainBaseline_Disp(dataDict, rtrnDict, assistDict):    
    polDict = dataDict['polDict']
    dartEnv = polDict['dartEnv']

    usePolicyArgsDict = defaultdict(int,{'rndPolicy':rtrnDict['rndPolicy'], 'renderRes':rtrnDict['renderRes']})   
    assistDict['passedMultNotFrc']=0
    paths = []
    path_bls =[]
    i=0
    #bounds from which to generate random values to use in optimization
    #can also use dartEnv.self.frcBnds
    assistMultBnds = dartEnv.getAssistBnds()
    #perform rollouts with specific assist values
    while i < rtrnDict['runIters']:
    #for i in range(iters):
        if(rtrnDict['setFrc']):
            rndAssist = dartEnv.getRandomAssist(assistMultBnds)
            dartEnv.setAssistDuringRollout(rndAssist, assistDict)
            #dartEnv.setForceMult(np.random.uniform(low=0.4, high=0.5),np.random.uniform(low=0.0, high=0.1))
        path, path_bl,_ = usePolicy(dataDict, optsDict=usePolicyArgsDict)#rtrnDict['rndPolicy'], renderRes=rtrnDict['renderRes'])
        #save path if either :
        #we're rejecting the bad ones and the path is longer than what is considered bad (we use  no bad rollouts and all good rollouts) OR
        #we're not rejecting bad ones, the path is 'bad' or rnd draw is less than "keep" rate setting (we use all bad and keepRate% of good rollouts)
        # if (rtrnDict['rejectBad'] and (len(path['returns']) > rtrnDict['badPathLen'])) or \
        #     (not rtrnDict['rejectBad'] and ((len(path['returns']) <= rtrnDict['badPathLen'] ) or \
        #                                      (np.random.uniform(low=0.0, high=1.0) <= rtrnDict['keepRate']))): 
        print("iter {}".format(i))
        i+=1
        paths.append(path)
        path_bls.append(path_bl)
    #fit baseline to new paths
    for i in range(rtrnDict['retrainIters']):
        polDict['baseline'].fit(paths)
    return polDict['baseline'], paths, path_bls
#render an image and write it to a file - run each frame
def renderAndSave(dartEnv, imgFN_prefix, i):
    import imageio
    img = dartEnv.render(mode='rgb_array')
    fileNum = "%04d" % (i,)
    fileName = imgFN_prefix + '_'+ fileNum + '.png'
    imageio.imwrite(fileName, img)  

#initialize for saving images - run 1 time per image sequence
def initImgSave(dartEnv, imgBaseDir='dartEnv_recdata/' ):
    dirName = dartEnv.getRunDirName()
    directory = os.path.join(os.path.expanduser( '~/{}'.format(imgBaseDir)) + dirName)
    if not os.path.exists(directory):
        os.makedirs(directory)
    imgFN_prefix = os.path.join(directory, dartEnv.getImgName())
    renderAndSave(dartEnv, imgFN_prefix,0)  
    return imgFN_prefix

#consume trained policy to see how it performs in env, using value function predictions 
#for force given each state observation
#follow this pattern :
#1) randomly set initial q/qdot of environment, use this state as feature to value func
#2) find vf-predicted optimal force for given state
#3) set force in environment using env.setAssistDuringRollout(frc)
#4) perform 1 rollout step
#repeat 2-4
#dataDict holds dicts : polDict,trainDict, expDict
#polDict : dictionary holding policy and baseline 
#trainDict : dictionary holding policy training info
#env : environment
#optsDict : dictionary of options for control : 'rndPolicy', 'renderRes','recording','debug','optOnlyBad' 
#rndPolicy : boolean whether or not to use a deterministic policy (policy mean) or stochastic
#renderRes : whether or not to render the results(Slower)
#if vfOptObj is none, then ignores optimization
#optsDict can hold ints, which will be interpretted as booleans or ints
#optsDict['findBotFrc'] : 
    #0 : ignore bot-generated force
    #1 : solve for bot for each new action
    #2 : solve for bot for each frame (requires new action for each frame as well, so env needs access to policy)
def usePolicy(dataDict, vfOptObj=None, numRndAssist=10, optsDict=defaultdict(int)):
    polDict = dataDict['polDict']
    trainDict = dataDict['trainDict']
    env = polDict['env']
    dartEnv = polDict['dartEnv']
    policy = polDict['policy']
    baseline = polDict['baseline']
    discount = trainDict['discount']
    doOptOfAll = (vfOptObj is not None) and not optsDict['optOnlyBad']
    doOptOnlyBad = (vfOptObj is not None) and optsDict['optOnlyBad']
    getEefFrcDicts = optsDict['getEefFrcDicts']
    pausePerStep = optsDict['pausePerStep']
    
    iterSteps =optsDict['iterSteps']
    if (iterSteps== 0):
        iterSteps = 300
    
    simEEFResDictAra = []

    #consume policy
    observations = []
    actions = []
    actionsUsed = []
    rewards = []
    if(not optsDict['rndPolicy']) and (optsDict['debug']):#deterministic policy - use mean
        print('Using Mean of policy')
    
    #process begins with reset
    observation = env.reset()
    #pass observation to find ordered dictionary of best assist proposals from value function, and modify observation to match this assist
    if doOptOfAll:
        observation = vfOptObj.setOptAssistForState(observation, numRndAssist)
    
    if(optsDict['renderRes']):
        imgFN_prefix=''
        if (optsDict['recording']):
            imgFN_prefix = initImgSave(dartEnv)
        else:
            env.render()
    iter = 0
    for i in range(iterSteps):
        # policy.get_action() returns a pair of values. The second one returns a dictionary, whose values contain
        # sufficient statistics for the action distribution (mean, log_std). 
        #i.e. : actionStats = dict(mean=mean, log_std=log_std)
        
        action, actionStats = policy.get_action(observation)
        if(not optsDict['rndPolicy']):#deterministic policy - use mean
            action = actionStats['mean']            
        
        # the last entry of the tuple stores diagnostic information about the environment. 
        next_observation, reward, terminal, infoDict = env.step(action)
        if pausePerStep : 
            print('paused for input')
            input()
        #actionsUsed might change if force from bot is used instead of constant, or if proposal is beyond action bounds
        #need to determine action - since policy is trained with frameskip span for each reward, 
        #action is mean actions used over entire frameskip, if action changes during frameskip 
        #(i.e. during optimization process this may happen (although doesn't have to))
        actUsed = infoDict['ana_actionUsed']
        #print('Pol mean action : {} \n\nAction Ana Used : {}\n are close : {}'.format(action, actUsed, np.allclose(actUsed, action)))        
        #action = actUsed#np.copy(infoDict['ana_actionUsed'])  
        #list of dictionaries of per-frame tuples of eef frc result
        if (getEefFrcDicts):
            simEEFResDictAra.append(dartEnv.getPerStepEefFrcDicts())
            
        observations.append(observation)
        actions.append(action)
        actionsUsed.append(actUsed)
        rewards.append(reward)
        
        #pass q/qdot to find ordered dictionary of best force proposals from value function, and modify observation to match this force
        if doOptOfAll:
            observation = vfOptObj.setOptAssistForState(next_observation, numRndAssist)
        else:
            observation = next_observation
        #print('reward : {:3.3f}'.format(reward))
        if(optsDict['renderRes']):
            if(optsDict['recording']):
                renderAndSave(dartEnv, imgFN_prefix,i+1)
            else:
                env.render()
        #print("obs : {}".format(observation[-2:]))
        iter = iter+1
        if terminal:
            #don't terminate bad rollouts - see if we can "jiggle" out of them - assume a rollout is bad if final step has reward  < some value
            #Currently ANA is setting reward to 0 on terminal step of bad rollouts.  
            #good rollout
            if (vfOptObj is None) or reward > dartEnv.getMinGoodRO_Thresh() :
                # Finish rollout if terminal state reached
                #if (optsDict['debug']):
                print('Terminal break after %s iters' %(str(iter)))
                break
            else : 
                #bad rollout due to low return, see if we can get a better force proposal
                if doOptOnlyBad:
                    observation = vfOptObj.setOptAssistForState(next_observation, numRndAssist)
                #retry with new force
                print('bad rollout attempting to recover')
    # We need to compute the empirical return for each time step along the
    # trajectory
    path = dict(
        observations=np.array(observations),
        actions=np.array(actions),
        actionsUsed = np.array(actionsUsed),
        rewards=np.array(rewards),
    )
    #perform final calculations on rollout for return, advantage, etc
    path, path_baseline = finalizeUsePolicy(path, baseline, discount)
    #return path, the baseline for the path, and the array of dictionaries of results (tuples) of force calcs at eef for ana and helper bot
    return path, path_baseline, simEEFResDictAra

#only called internally - finalize results from rollout
#take path (dict holding observations, actions and observations of rollout ) and calculate per step returns and advantages
def finalizeUsePolicy(path, baseline, discount):
    rewards = path['rewards']
    #use trained baseline to predict value function
    path_baseline = baseline.predict(path)
    advantages = []
    returns = []
    return_so_far = 0
    #step back in time to build advantage and returns
    for t in range(len(rewards) - 1, -1, -1):
        return_so_far = rewards[t] + discount * return_so_far
        returns.append(return_so_far)
        advantage = return_so_far - path_baseline[t]
        advantages.append(advantage)
    # The advantages and returns are stored backwards in time, so we need to reverse them
    advantages = np.array(advantages[::-1])
    returns = np.array(returns[::-1])
    #normalizing advantages to avoid problems from reward scaling
    advantagesNorm = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    
    path['advantages'] = advantagesNorm
    path['returns'] = returns
    #get estimate of value function -> rewards[i] - unnormalized advantages[i]
    path['advantages_raw'] = advantages

    return path, path_baseline


#terminate passed environment
def killEnv(env):
    env.terminate()
    env.render()
#perform this over a list of rollout reward results
def showRes(dispStr, itr, tot_rwds):
    avg_return = np.mean(tot_rwds)
    var_return = np.var(tot_rwds)
    disp = var_return/avg_return
    return avg_return, var_return, disp
    
def buildPlotTitle(expDict, plotType):
    if(1==plotType):#returns
        title='Results for Force vs. Returns : TRPO policy Arch : '
    elif(2==plotType):#rollout duration
        title='Results for Force vs. Rollout Duration : TRPO policy Arch : ' 
    else:
        title='Unknown plot type {} : TRPO policy Arch : '.format(str(plotType))
    
    title = title + ','.join([str(x) for x in expDict['polNetArch']])
    title = title + ' and ' + expDict['blType'] + ' baseline '
    if(expDict['blType'] == 'MLP'):
        title = title +' Arch : ' + ','.join([str(x) for x in expDict['blMlpArch']])
    return title

def calcMeanVarDisp(frcDict):
    xVals = []
    means = []
    medians = []
    variances = []
    dispersions = []
    res = {}
    
    for frc in frcDict:
        valNPAra = np.array(frcDict[frc])
        xVals.append(frc)
        mean = np.mean(valNPAra)
        med = np.median(valNPAra)
        var = np.var(valNPAra)
        means.append(mean)
        medians.append(med)
        variances.append(var)
        dispersions.append(var/mean)
    res['xVals'] = xVals
    res['means'] = means
    res['medians'] = medians
    res['variances']= variances
    res['dispersions']=dispersions
    return res

#build plot for exp results held in fileName
def plotRes1Dof(filename, polDict, trainDict, title, plotType, tmpData=None):
    #data is in format of list of tuples
    #idx 0 is frcmult (force applied is frcmult * mg @ pi/4 +x, +y)
    #idx 1 is dictionary of results with keys 'advantages' and 'rewards'
    #each value is a list of advantages and rewards at timesteps t-idx
    #
    
    if None == tmpData :
        tmpData=readTestResults(filename)
        
    #plot results
    plotVals = []
    minRetTup = (-1, 100000)
    maxRetTup = (-1, -minRetTup[1])
    frcDict = dict()
       #plot type : 1 = frcmult x returns
    if (1==plotType):
        elemKey = 'returns'
        legendVal = 'returns'
        yLabelStr = 'Final return value'
    elif (2==plotType):#duration == use length of elem[1][elemKey]
        elemKey = 'returns'#anything will work here
        legendVal = 'durations'
        yLabelStr = 'Rollout Duration1'
#    elif (3==plotType):#rewards - advantages == baseline val == value function
#        yLabelStr = 'Value function'
#        legendVal = 'baseline eval'
        

    for elem in tmpData:
        frc = elem[0]
        if(2==plotType):#length
            ret = len(elem[1][elemKey])
        #elif(3==plotType):
        else:
            ret = elem[1][elemKey][0]
        if(ret < minRetTup[1]):
            minRetTup = (frc,ret)
        elif(ret > maxRetTup[1]):
            maxRetTup = (frc,ret)   
        plotVals.append((frc, ret))
        if frc not in frcDict :
            frcDict[frc] = []        
        frcDict[frc].append(ret)
    #frcDict has dictionary of all y values for given frc multiplier

    #main results    
    xAxis, yAxis = zip(*plotVals)  
    pd = calcMeanVarDisp(frcDict)
    
    
    
    #colors = ['b', 'r', 'g', 'c', 'm']
    pts = plt.scatter(xAxis, yAxis, c='b', s=1)
    meds = plt.scatter(pd['xVals'],pd['medians'],c='r',s=10)
    mus  = plt.scatter(pd['xVals'],pd['means'],c='g',s=10)#more affected by outliers
    plt.legend((pts, meds, mus),
               (legendVal, 'Medians', 'Means'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.title(title)
    plt.xlabel('Assist force modifier ( X mg)')
    plt.ylabel(yLabelStr)
    plt.show()
    
  
#plot heatmap of resMat data(2d array) using vList as x and y axes
def HMPlot(vList, resMat, pltTitle, pltIdx, numTtlPlots=2):
    #put all plots on same row    
    plt.subplot(1, numTtlPlots, pltIdx)
    plt.title(pltTitle)
    plt.pcolor(vList,vList,resMat)
    plt.colorbar()
    
#return tuple with returns[0], bl[0], returns[idxOfMaxBL], bl[idxOfMaxBL],returns[idxOfMaxRet], bl[idxOfMaxRet]
def getIndivReturnAndBLPred(dataDict):
    import operator
    expDict = dataDict['expDict']
    polDict = dataDict['polDict']
    usePolicyArgsDict = defaultdict(int,{'rndPolicy':expDict['useRndPol']})
    #usePolicyArgsDict = {'rndPolicy':expDict['rndPolicy'], 'renderRes':False, 'recording':False, 'debug':False, 'optOnlyBad':True}    
    
    path,_,_ = usePolicy(dataDict, optsDict=usePolicyArgsDict)
    tmpPath = dict(
        observations=np.array(path['observations'])
    )
    path_baseline = polDict['baseline'].predict(tmpPath)

    idxRet, valRet = max(enumerate(path['returns']), key=operator.itemgetter(1))   
    idxBL, valBL = max(enumerate(path_baseline), key=operator.itemgetter(1))
    
    #tuple with return and baseline prediction at different steps in rollout
    #step 0, step w/max return value, step with max value pred
    retVal = (path['returns'][0],path_baseline[0], idxRet, valRet, path_baseline[idxRet], idxBL, path['returns'][idxBL],valBL)
    
    return retVal


#compare returns to baseline predictions at step 0 and at step[bl_max] and step[returns_max]
def testMaxBLAgainstRet(dataDict):
    expDict = dataDict['expDict']
    numTests = 50
    resDict = runTestsRetAndBLPred(numTests, dataDict)
    
#    resDict['stepMu']=stepMu
#    resDict['stepStd']=stepStd
#    resDict['stepName']=stepName
#    resDict['resRet']=resRet
#    resDict['resBL']=resBL
    #plt.figure(1)
    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    xAxis = np.linspace(0,numTests,numTests,endpoint=False)
    pltLoc = 311
    expName = getExpNameDisp(expDict)
    for i in range(3):        
        plt.subplot((pltLoc+i))        
        plt.plot(xAxis,resDict['resRet'][i],c='r', label='Ret @{}'.format(resDict['stepName'][i]))
        plt.plot(xAxis,resDict['resBL'][i],c='b', label='BL @{}'.format(resDict['stepName'][i]))

#       plt.legend(loc='lower left',ncol=4,fontsize=8)
        plt.title('Compare Step {}={}(std {}) Returns and BL Preds for {}'.format(resDict['stepName'][i],resDict['stepMu'][i],resDict['stepStd'][i],expName),fontsize=10)
        plt.xlabel('Rollout #')
        plt.ylabel('Return/BL Pred')
#    plt.subplot(311)
#    plotIndivPlt(plt, numTests, 0, resDict)
#    plt.subplot(312)
#    plotIndivPlt(plt, numTests, 1, resDict)
#    plt.subplot(313)
#    plotIndivPlt(plt, numTests, 2, resDict)
    plt.show()
#run numTests iterations of tests, accumulate results into dictionary
def runTestsRetAndBLPred(numTests, dataDict):
    resRet0 = []
    resBL0 = []
    idxsMaxRet = []
    idxsMaxBL = []
    resRetMaxRet = []
    resBLMaxRet = []
    resRetMaxBL = []
    resBLMaxBL = []
    for i in range(numTests):
        tmpTpl = getIndivReturnAndBLPred(dataDict)        
        resRet0.append(tmpTpl[0])
        resBL0.append(tmpTpl[1])
        idxsMaxRet.append(tmpTpl[2])
        resRetMaxRet.append(tmpTpl[3])
        resBLMaxRet.append(tmpTpl[4])
        idxsMaxBL.append(tmpTpl[5])        
        resRetMaxBL.append(tmpTpl[6])
        resBLMaxBL.append(tmpTpl[7])
    
    stepMu = []
    stepStd = []
    stepName=[]
    stepMu.append(0)
    stepMu.append(round(np.mean(idxsMaxRet)))
    stepMu.append(round(np.mean(idxsMaxBL))) 
    
    stepStd.append(0)
    stepStd.append(round(np.std(idxsMaxRet)))
    stepStd.append(round(np.std(idxsMaxBL))) 
    
    stepName.append('@IDX 0')
    stepName.append('@IDX of Max Ret')
    stepName.append('@IDX of Max Val Pred')
    
    resRet = []
    resRet.append(resRet0)
    resRet.append(resRetMaxRet)
    resRet.append(resRetMaxBL)
    
    resBL=[]    
    resBL.append(resBL0)
    resBL.append(resBLMaxRet)
    resBL.append(resBLMaxBL)   
    
    resDict = {}
    resDict['stepMu']=stepMu
    resDict['stepStd']=stepStd
    resDict['stepName']=stepName
    resDict['resRet']=resRet
    resDict['resBL']=resBL
    return resDict

#build an experiment dictionary and directory name/prefix based on passed experiment descriptors for expLite experiment
def buildExpLiteDict(rwdDigsToUse, envName, itrs, blArch, polArch, numBatches, maxPathLength, useCG=True, gae_lambda=1.0, optimizerArgs=None, resumeExp=False, isNorm=False, resumeExpDict={},retrain=False):   
    #variables defining experiment - dictionary is passed to expLite as variant
    expDict = dict()
    # number of iterations to train
    expDict['numIters'] = itrs
    expDict['blType'] = 'MLP'
    expDict['blMlpArch'] = blArch
    #environment name
    expDict['envName'] = envName
    expDict['isGymEnv'] = True
    expDict['polNetArch'] = polArch
    expDict['numBatches'] = numBatches
    #format of reward being used
    expDict['rwdCmpFormat'] = rwdDigsToUse
    #maximum allowed path length = defaults to 1k, but smaller is also sufficient
    expDict['maxPathLength']=maxPathLength
    #whether to use trpo or ppo
    expDict['useCG']=useCG
    #gae_lambda - defaults to 1 -> default TRPO.  
    # set to be 0<v<1 - seems to only diminish discount factor by gae_lambda for -advantages- calculation in base sampler code (i.e. discount is deeper) see line 63 in sampler/base.py
    expDict['gae_lambda']=gae_lambda
    #if optimizer args exist, use them
    expDict['optimizerArgs']=optimizerArgs
    #normalized env forces all outputs to be betwee +/- action_space values (And does so poorly, assumes output from net is +/- 1)
    expDict['isNormalized'] = isNorm
    
    #dir name prefix encodes all rewards used in training
    expPrefixSt = 'exp-{}'.format(rwdDigsToUse)
    useTRPOStr = 'TRPO' if expDict['useCG'] else 'LBFGS-NPO'
    gaeStr = '' if expDict['gae_lambda'] == 1.0 else 'GAE-lam-{}'.format(expDict['gae_lambda'])
    normStr = '' if expDict['isNormalized'] else 'unNormEnv'
    expDict['expPrefix'] = '_'.join([expPrefixSt,
           'bl','-'.join([str(x) for x in expDict['blMlpArch']]),
           'pl','-'.join([str(x) for x in expDict['polNetArch']]),
           'it',str(expDict['numIters']), 'mxPth',str(expDict['maxPathLength']),
           'nBtch',str(expDict['numBatches']),'ALG', useTRPOStr, gaeStr, normStr])
        
    if resumeExp :
        from rllab import config
        #for resuming training - otherwise set resumeData to None - also passes # of iterations to extend training beyond current training to
        #bypass rllab bug
        resume_from = config.LOG_DIR + '/local/' +resumeExpDict['prfx']+'/'+resumeExpDict['dir']+'/params.pkl'
        expDict['resumeData'] = resume_from+'&|&'+str(expDict['numIters'])+'&|&' + str(expDict['numBatches'])
    else :
        expDict['resumeData'] = ''
    return expDict

#convert a given experiment directory name as devined in trpoTests_ExpLite o a descriptive string
def expDirNameToString(expDirName, descWidth=100):
    import gym.envs.dart.dart_env_2bot as dartEnv2bot    
    import textwrap
    #expDirName = 'exp-011101110111_bl_32-32-16_pl_64-32-32_it_4051_mxPth_500_nBtch_200000_ALG_TRPO__2018_09_16_06_11_31_0001'
    valsList = expDirName.split('_')
    rwdDigs = valsList[0].split('-')[-1]
    rwdVals = dartEnv2bot.DartEnv2Bot.bldRwdListFromBinDig(rwdDigs)
    rwdValStr = 'Reward Funcs : {}'.format(rwdVals)
    
    dataDict = {}        
    dataDict['rwds'] = rwdValStr
    #bl arch is idx 2; pol archi is idx 4; 
    dataDict['blArch']='({})'.format(','.join(valsList[2].split('-')))
    dataDict['polArch']='({})'.format(','.join(valsList[4].split('-')))
    dataDict['numIters']=valsList[6]
    dataDict['maxPathLen']=valsList[8]
    dataDict['batchSize']=valsList[10]
    dataDict['alg']=valsList[12]
    if len(valsList[13]) > 0 :#gae lambda tag present - modifies size of valsList
        dataDict['gae'] = valsList[13].split('-')[-1]
    else : 
        dataDict['gae'] = '1.0'    
    dataDict['normedEnv']=False if 'unnormenv' in valsList[14] else True
    expNameList = textwrap.wrap('Alg : {} Rwds : {} Pol Arch : {} | BL Arch : {} | # Epochs : {} | Batch Size : {} | Normed Env : {}'.format(valsList[12], rwdValStr,dataDict['polArch'], dataDict['blArch'],dataDict['numIters'],dataDict['batchSize'], dataDict['normedEnv']), width=descWidth)
    
    if len(expNameList) > 1:
        expName = '\n'.join(expNameList)
    else:
        expName = expNameList[0]
        
    shortName = 'RWD:{}|P:{}|BL:{}|Iters:{}|Smpls:{}|{}'.format(rwdDigs, dataDict['polArch'], dataDict['blArch'],dataDict['numIters'],dataDict['batchSize'], ('N' if dataDict['normedEnv'] else 'U'))
    print('shortName : {}'.format(shortName))
    
    #print('expName : {}'.format(expName))
    return dataDict, expName, shortName


#Returns a function that maps n distinct indices to a different color - name is colormap
def getClrMap(n, name='hsv'):
    return plt.cm.get_cmap(name, (n+1))

#plot data from dataDict dict of lists, using keysToPlot to determine which keys to plot
#dataDict : dictionary of data
#keysToPlot - list of keys in dataDict to plot on same graph, contains a list per plot to be put in same plot
#exp title is name used in plot
def plotPerfData(expTitle, dataToPlotDict, keysToPlotList=None, yAxisScale='linear', clipYAtZero=True):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec    
    #if no specific keys passed, plot all data values
    if keysToPlotList is None :
        keysToPlotList = list(list(dataToPlotDict.keys()))
        
    numIndivPlots = len(keysToPlotList)
    #manage plotting different policy data side by side
    
    if (numIndivPlots < 3):#handle 
        cols = 1
        rows = numIndivPlots   
    else :        
        cols = 2
        rows = numIndivPlots//cols + 1
    gs = gridspec.GridSpec(rows, cols)
    gs.update(hspace=0.4)    
    numXVals = len(dataToPlotDict[keysToPlotList[0][0]]) 
     
    fig = plt.figure()
    fig.suptitle('{}\nTraining Plot over {} iters of :'.format(expTitle,numXVals))
    #fig2 = plt.figure(num=2, figsize=figsize)
    for i in range(numIndivPlots):
        keysToPlot = keysToPlotList[i]
        numXVals = len(dataToPlotDict[keysToPlot[0]]) 
        xAxis = np.linspace(0,numXVals,numXVals,endpoint=False)
        col = (i // rows)
        row = i % rows
        ax = fig.add_subplot(gs[row, col])  
        buildIndivPerfPlot(ax, dataToPlotDict, keysToPlot, xAxis, yAxisScale,clipYAtZero)    
        
        
#load multiple policies described by different expDicts, do not instance any environments 
def loadMultPolicies(polDirAra, envName, normEnv):
    polDicts = []
    expDicts = []
    for dataDirName in polDirAra : 
        dataDict = buildAllExpDicts(dataDirName=dataDirName, envName=envName, isNormalized=normEnv,useNewEnv=False, launchEnv=False)
        expDicts.append(dataDict['expDict'])
        polDicts.append(dataDict['polDict'])    
    return polDicts, expDicts

#build plot comparing multiple policies
def plotPolsSideBySide(polDirAra, envName, normEnv, keysToPlotList, yAxisScale='linear', clipYAtZero=True):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec 

    polDictAra, expDictAra = loadMultPolicies(polDirAra, envName, normEnv)  
    numPols = len(polDictAra)
    numIndivPlots = len(keysToPlotList)
    #manage plotting different policy data side by side
    cols = numPols
    rows = numIndivPlots
#    gs = gridspec.GridSpec(rows, cols)
#    gs.update(hspace=0.4)    
    col = 0
    fig, axes = plt.subplots(rows,cols)
    #fig = plt.figure()    
    fig.suptitle('Comparison of {} policies'.format(numPols))
   
    for expDict in expDictAra :
        _, expDesc, shortName = expDirNameToString(expDict['ovrd_dataDirName'],descWidth=70)
        dataDict = expDict['trainingInfoDict']            
      
        for i in range(numIndivPlots):
            keysToPlot = keysToPlotList[i]
            numXVals = len(dataDict[keysToPlot[0]]) 
            xAxis = np.linspace(0,numXVals,numXVals,endpoint=False)
            row = i
            ax = axes[row,col]
            if(i==0):
                ttlPrefix = '{}\nTraining Plot over {} iters of :'.format(expDesc,numXVals)
            else:
                ttlPrefix=''
            buildIndivPerfPlot(ax, dataDict, keysToPlot, xAxis, yAxisScale,clipYAtZero,ttlPrefix)    
        col +=1
        



#returns a solution to the polynomial parameterized equation at x given by coeffs
#where coeffs is deg+1(rows) x 1 array of coefficients for deg polynomial
def solvePolyFromCoeffs(coeffs, x):
    
    calcVal = 0
    numCoeffs = len(coeffs)
    #solve polynomial
    for i in range (numCoeffs - 1):
        calcVal += coeffs[i] * x
    #don't forget constant term       
    calcVal += coeffs[-1]
    return calcVal        

#plot policies on same set of axis
#normalizePerSmpl : rescale x so that it reflects # of samples, and not # of epochs
def plotPolsTogether(polDirAra, envName, normEnv, keysToPlotList, normalizePerSmpl=False, yAxisScale='linear', clipYAtZero=True):
    polDictAra, expDictAra = loadMultPolicies(polDirAra, envName, normEnv)  
    numPols = len(polDictAra)
    numIndivPlots = len(keysToPlotList)
    
    import matplotlib.pyplot as plt
    
    #manage plotting different policy data side by side
    cols = 1
    rows = numIndivPlots

    fig, axes = plt.subplots(rows,cols)
    #fig = plt.figure()    
    fig.suptitle('Comparison of {} policies on same graph:'.format(numPols))
    #build new data dict with keys for each policy
        
    #if normalizing per sample, then build eq with x:sample and y:value - this is to compare results from larger batches to smaller batches
    
    #use values directly, or coeffs keys in dataDict - find max x value that is represented in each data sample
    #using coefficients weights each sample equally, vs comparing per epoch
    maxXValToUse = 100000000000000
    for expDict in expDictAra :
        dataDict = expDict['trainingInfoDict']
        if normalizePerSmpl: 
            maxXVal = dataDict['xVec'][-1]
        else :
            maxXVal = len(dataDict['xVec'])
        
        if (maxXValToUse > maxXVal):
            maxXValToUse = maxXVal
            numXVals = len(dataDict['xVec'])
        
    xAxis = np.linspace(0,maxXValToUse,num=numXVals,endpoint=True)
    if normalizePerSmpl: 
        nXAxis = []
        for i in range(len(expDictAra)) :
            expDict = expDictAra[i]
            dataDict = expDict['trainingInfoDict']
            if normalizePerSmpl: 
                nXAxis.append(2*xAxis/dataDict['xVec'][-1] - 1)

    dataDictsToPlot=[]
    keysToPlotAra =[]  
    titlesPerPlotAra=[]
    for pltKeys in keysToPlotList:
        ttlDataDict = {} 
        keysToPlot =[]
        titlesPerPlot=[]
        for i in range(len(expDictAra)) :
            expDict = expDictAra[i]
            _, expDesc, shortName = expDirNameToString(expDict['ovrd_dataDirName'],descWidth=70)
            dataDict = expDict['trainingInfoDict']  
            for k,v in dataDict.items():
                if normalizePerSmpl:                    
                    kVals = k.split('_')
                    if len(kVals) > 1 and kVals[0] in pltKeys:
                        newKey='{}_{}'.format(shortName,kVals[0])
                        keysToPlot.append(newKey)
                        ttlDataDict[newKey]=np.polynomial.legendre.legval(nXAxis[i], v)#solvePolyFromCoeffs(v, xAxis)
                else :
                    if k in pltKeys:
                        print('{} in pltKeys'.format(k))
                        newKey='{}_{}'.format(shortName,k)
                        keysToPlot.append(newKey)
                        ttlDataDict[newKey]=v[:maxXValToUse]
        
        for k in pltKeys : 
            titlesPerPlot.append(k)
        dataDictsToPlot.append(ttlDataDict)
        keysToPlotAra.append(keysToPlot)
        titlesPerPlotAra.append(titlesPerPlot)

            
    for i in range(len(dataDictsToPlot)):
        dataDict = dataDictsToPlot[i]
        keysToPlot = keysToPlotAra[i]
        titlesPerPlot = titlesPerPlotAra[i]
        row = i
        ax = axes[row]
        buildIndivPerfPlot(ax, dataDict, keysToPlot, xAxis, yAxisScale,clipYAtZero,titlesPerPlot=titlesPerPlot)    
    
    return expDictAra

def _formatSubplotTitle(ax, numPlotVals, ttlPrfx, titlesPerPlot):
    if(numPlotVals > 10):
        numHalf = numPlotVals//2
        ttl1 = ', '.join(titlesPerPlot[:numHalf])
        ttl2 = ', '.join(titlesPerPlot[numHalf:])
        titleKeys = ',\n'.join([ttl1, ttl2])
    else :            
        titleKeys = ','.join(titlesPerPlot)
    if(len(ttlPrfx) > 0):
        titleKeys = '{}\n{}'.format(ttlPrfx,titleKeys)
    print('Title : {}---'.format(titleKeys))
    ax.title.set_text('{}'.format(titleKeys))
    
#build individual subplot for passed data dictionary over keys in keysToPlot list
#ax is an axes object
def buildIndivPerfPlot(ax, dataDict, keysToPlot, xAxis, yAxisScale, clipYAtZero, ttlPrfx='', titlesPerPlot=[]):    
    numPlotVals = len(keysToPlot)
    clrMap = getClrMap(numPlotVals)
    if (len(titlesPerPlot) > 0):
        _formatSubplotTitle(ax, numPlotVals, ttlPrfx, titlesPerPlot)
    else :
        _formatSubplotTitle(ax, numPlotVals, ttlPrfx, keysToPlot)
        
    if('symlog' in yAxisScale):
        ax.set_yscale('symlog', linthreshy=0.01)
    else :
        ax.set_yscale(yAxisScale)    
    for i in range(numPlotVals):
        key = keysToPlot[i]   
        ydata =  np.array(dataDict[key])
        if clipYAtZero :
            y=np.clip(ydata, a_min=0, a_max=None)
        else:
            y=ydata
        ax.plot(xAxis, y, color=clrMap(i))
    ax.legend(keysToPlot)
    
    
    
    
#def plotIndivPlt(plt, numTests, idx, resDict,expDict):
#    xAxis = np.linspace(0,numTests,numTests,endpoint=False)
#    plt.plot(xAxis,resDict['resRet'][idx],c='r', label='Ret @{}'.format(resDict['stepName'][idx]))
#    plt.plot(xAxis,resDict['resBL'][idx],c='b', label='BL @{}'.format(resDict['stepName'][idx]))
#
##    plt.legend(
##           loc='lower left',
##           ncol=4,
##           fontsize=8)
#    plt.title('Compare Step {}={}(std {}) Returns and BL Preds for {}'.format(resDict['stepName'][idx],resDict['stepMu'][idx],resDict['stepStd'][idx],getExpNameDisp(expDict)),fontsize=10)
#    plt.xlabel('Rollout #')
#    plt.ylabel('Return/BL Pred')


#################################################
# fundamental tests of basic functionality
#################################################

#test performance of passed policy - poll random states from env, feed to policy, show resultant actions 
def testPolicyOnRandomStates( dataDict, numStatesToTest=100):
    dartEnv = dataDict['dartEnv']
    policy = dataDict['polDict']['policy']
    #get all limits of observation variables
    skelValLims = dartEnv.activeRL_SkelHndlr.getObsLimits()
    ttlActions = [None]*numStatesToTest
    ttlMeanActions = [None]*numStatesToTest

    ttlObs = [dartEnv.getRandomObservation() for i in range(numStatesToTest)]
    i = 0
    for obs in ttlObs:
        action, actionStats = policy.get_action(obs)
        ttlActions[i]=action
        ttlMeanActions[i]=actionStats['mean']
        i+=1
    res = {}
    res['skelValLims']=skelValLims
    res['obs']=np.array(ttlObs)
    #actions sampled from pol dist at given state
    res['actions']=np.array(ttlActions)
    #mean of pol dist at given state
    res['meanActions']=np.array(ttlMeanActions)
    #number of states
    res['numStatesTested']=numStatesToTest

    return res

#test performance of passed policy using passed paths from rollouts 
#paths is list of path dictionaries from rollouts
def testPolicyOnROStates(dataDict, paths):
    dartEnv = dataDict['dartEnv']
    policy = dataDict['polDict']['policy']
    #paths is list of paths from rollouts
    #get all limits of observation variables
    skelValLims = dartEnv.activeRL_SkelHndlr.getObsLimits()
    numStatesToTest = 0
    ttlActions = []
    ttlActionsFromRO=[]
    ttlMeanActions =[]
    ttlObs = []
    eqAction = 0
    closeAction = 0
    farAction = 0
    for path in paths:
        pathLen = len(path['actions'])
        numStatesToTest += pathLen
        for i in range(pathLen):
            obs = path['observations'][i]
            actFromPolTest = path['actions'][i]
            ttlActionsFromRO.append(actFromPolTest)
            action, actionStats = policy.get_action(obs)
            meanAction = actionStats['mean']

            if(np.array_equal(actFromPolTest, meanAction)):
                eqAction +=1
            elif (np.allclose(actFromPolTest, meanAction)):
                closeAction+=1
            else:
                farAction +=1
            ttlActions.append(action)
            ttlMeanActions.append(meanAction)
            ttlObs.append(obs)

    res = {}
    res['skelValLims']=skelValLims
    res['obs']=np.array(ttlObs)
    #actions sampled from pol dist at given state
    res['actions']=np.array(ttlActions)
    #mean of pol dist at given state
    res['meanActions']=np.array(ttlMeanActions)
    #mean actions used in rollout- have been "normalized"
    res['meanActionsRO']=np.array(ttlActionsFromRO)
    #number of states
    res['numStatesTested']=numStatesToTest
    #compare actions from deterministic policy queries - should -all- be equal
    res['actionSimilarity']={'eqAction':eqAction, 'closeAction':closeAction, 'farAction':farAction}

    return res



#plot min/max/mean/std of lists of vectors
#x is list of vector values of indeterminate length
#minCol and maxCol are 
def plotTestValsAvgStdMinMax(xRaw, colSlice, title, lims=None):
    import matplotlib.pyplot as plt
    x = xRaw[:,colSlice]
    mins = np.amin(x, axis=0)
    maxes = np.amax(x, axis=0)
    means = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    numSamples = len(x)
    xVals = np.arange(len(mins))
    
    maxY= np.max(maxes)
    minY= np.min(mins)
    absMinY = np.abs(minY)
    if lims is None:
        adj = max(.5, .1*absMinY)
        lims = [minY-adj, maxY+adj]
    print('min y : {} adj : {} used min : {} max y : {}'.format(minY, adj, lims[0],lims[1]))
    # create stacked errorbars:
    plt.ylim(lims[0],lims[1])
    plt.title('{} min/max/mean/std plot for dof  cols {} to {} over {} samples'.format(title,colSlice.start, colSlice.stop, numSamples))
    plt.errorbar(xVals, means, std, fmt='ok', lw=3)
    plt.errorbar(xVals, means, [means - mins, maxes - means], fmt='.k', ecolor='gray', lw=1)
    

    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.show()
    
    
#for observation space
def plotTestObsAndLims(obsRaw, obsLims, colSlice):
    import matplotlib.pyplot as plt
    obs = obsRaw[:,colSlice]
    mins = np.amin(obs, axis=0)
    maxes = np.amax(obs, axis=0)
    means = np.mean(obs, axis=0)
    std = np.std(obs, axis=0)
    numSamples = len(obs)
    xVals = np.arange(len(mins))
    
    maxY= np.max(maxes)
    minY= np.min(mins)
    absMinY = np.abs(minY)
    # create stacked errorbars:
    plt.ylim((minY-.1*absMinY), 1.1*maxY)
    plt.title('Observation min/max/mean/std plot for dof cols {} to {} over {} samples with enforced limits drawn as lines'.format(colSlice.start, colSlice.stop, numSamples))
    plt.errorbar(xVals, means, std, fmt='ok', ecolor='red', lw=3)
    plt.errorbar(xVals, means, [means - mins, maxes - means],fmt='.k', ecolor='gray', lw=1)
    
    minJnts = obsLims['obs_lowBnds'][colSlice]
    maxJnts = obsLims['obs_highBnds'][colSlice]
    plt.plot(xVals, minJnts)
    plt.plot(xVals, maxJnts)


    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.show()
    
   
