# -*- coding: utf-8 -*-
import os, sys
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
sys.path.insert(0, os.path.expanduser( '~/rllab_project1/'))
import trpoLibFuncs as tFuncs
#trpo 
#import rllab.misc.logger as logger
#from rllab.misc.ext import set_seed
#!!!! do not call from spyder - stderr clobbers root partition on JT's machine, can't remap output to different dir
#from rllab.algos.trpo import TRPO
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.algos.npo import NPO
#for configuration values
from rllab import config
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from rllab.misc.instrument import run_experiment_lite

#to be able to modify global value for using simple reward
import gym.envs.dart.dart_env_2bot as dartEnv2bot

def run_task(v):
    expDict=v
    ###############################
    #Env    
    if(expDict['isNormalized']):
        if(expDict['isGymEnv']):
            env = normalize(GymEnv(expDict['envName'],record_video=False, record_log=False))
        else :
            env = normalize(expDict['envName'])
        #if env is normalized then it is wrapped
        #dartEnv = env.wrapped_env.env.unwrapped
    else :#if not normalized, needs to be gym environment
        env = GymEnv(expDict['envName'],record_video=False, record_log=False)

    if (expDict['blType'] == 'linear'):
        bl = LinearFeatureBaseline(env_spec=env.spec)
    elif (expDict['blType'] == 'MLP'):
        #use regressor_args as dict to define regressor arguments like layers 
        regArgs = dict()
        regArgs['hidden_sizes'] = expDict['blMlpArch']
        #only used if adaptive_std == True
        regArgs['std_hidden_sizes']= expDict['blMlpArch']
        #defaults to normalizing
        regArgs['normalize_inputs'] = False
        regArgs['normalize_outputs'] = False
        #regArgs['adaptive_std'] = True
        #regArgs['learn_std']= False  #ignored if adaptive_std == true - sets global value which is required for all thread instances
        bl = GaussianMLPBaseline(env_spec=env.spec, regressor_args=regArgs)
    else:
        print('unknown baseline type : ' + expDict['blType'])
        bl = None
    
    ###############################
    #Policy
    pol = GaussianMLPPolicy(env_spec=env.spec,        
        hidden_sizes=expDict['polNetArch']#must be tuple - if only 1 value should be followed by comma i.e. (8,)
    )   

    ###############################
    #RL Algorithm

    #allow for either trpo or ppo
    optimizerArgs = expDict['optimizerArgs']
    if optimizerArgs is None: optimizerArgs = dict()

    if expDict['useCG'] :
    #either use CG optimizer == TRPO
        optimizer = ConjugateGradientOptimizer(**optimizerArgs)
        print('Using CG optimizer (TRPO)')
    #or use BFGS optimzier -> ppo? not really
    else:
        optimizer = PenaltyLbfgsOptimizer(**optimizerArgs)
        print('Using LBFGS optimizer (PPO-like ?)')
    #NPO is expecting in ctor : 
    #self.optimizer = optimizer - need to specify this or else defaults to PenaltyLbfgsOptimizer
    #self.step_size = step_size : defaults to 0.01
    #truncate_local_is_ratio means to truncate distribution likelihood ration, which is defined as
    #  lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
    # if truncation is not none : lr = TT.minimum(self.truncate_local_is_ratio, lr)        
    #self.truncate_local_is_ratio = truncate_local_is_ratio
    algo = NPO(
        optimizer=optimizer, 
        env=env, 
        policy=pol, 
        baseline=bl, 
        batch_size=int(expDict['numBatches']), 
        whole_paths=True, 
        gae_lambda=float(expDict['gae_lambda']), 
        max_path_length=int(expDict['maxPathLength']),
        n_itr=int(expDict['numIters']),
        discount=0.99,
        step_size=0.01,
        start_itr=1)

    algo.train()

#using lists so that can automate multiple runs - TODO use file-based experimental configs instead
seeds = [99,1099,75757,132554,245759]
blArchList = [(8,8), (16,16), (32,32), (64,64),  (128,128), (8,8,8), (16,16,8), (32,16,8), (32,32,16), (128, 64, 32), (100,50,25), (128,128)]
polArchList = [(64,64),(96,32),(96,96),(128,128),(64,32,32),(64,64,32), (128, 64, 64), (100,50,25), (256,64) ]
itersList = [50,100,150,200,250,500, 1000, 5000, 20000]

#if running this crashes or leaves windows open use this cmd to clear orphaned processes: 
#ps aux | grep /home/john/rllab/scripts/run_experiment_lite.py | awk '{print $2}' | xargs kill 
#ps aux | grep /home/jturner65/rllab/scripts/run_experiment_lite.py | awk '{print $2}' | xargs kill 

#seed idx
sd=0
#baseline approx
bl=2
#gae paper arch is idx -2; 
pl=-3
#200 iters is i==3, 1000 is -2
i=-1
numBatches = 200000# 50000#200000
#max allowed path length - 2000 was used in GAE paper
maxPathLength = 500#300

#use conjugate gradient optimizer or LBFGS optimizer - using cg is what TRPO does = this TRUE means use TRPO
useCG=True
#gae_lambda - defaults to 1 -> default TRPO.set to be 0<v<=1 - diminish discount factor by gae_lambda for -advantages- calculation in base sampler code (i.e. discount is deeper) see line 63 in sampler/base.py
gae_lambda=1.0
#custom optimizer Args, if any are used
optimizerArgs=None

#3d env getting up with kinematic robot
#envName = 'DartStandUp3d_2Bot-v1'
#3d env getting up with constraint connected to ANA
envName = 'DartStandUp3d_2Bot-v2'
#gae env name
#envName = 'DartStandUp3d_GAE-v1'
#whether we use normalized env or not
isNorm = True

#what reward function components to train on
#rwdList = dartEnv2bot.DartEnv2Bot.rwdCompsUsed
#rwdList = ['eefDist','action','height','lFootMovDist','rFootMovDist','comcop','UP_COMVEL','X_COMVEL','Z_COMVEL']
#below is rwdList that will use knee action and goal pose matching to help train, and will penalize any assist force coming from the constraint
rwdList=['eefDist','action','height','lFootMovDist','rFootMovDist','comcop','UP_COMVEL','X_COMVEL','Z_COMVEL','kneeAction','matchGoalPose']#,'assistFrcPen']
#rwdList=['action','height','lFootMovDist','rFootMovDist','comcop','UP_COMVEL','X_COMVEL','Z_COMVEL','kneeAction','matchGoalPose','assistFrcPen']
#
#rwdList must be some combination of string names of implemented reward components : rwdNames = ['eefDist','action', 'height','footMovDist','lFootMovDist','rFootMovDist','comcop','contacts','UP_COMVEL','X_COMVEL','Z_COMVEL','GAE_getUp','kneeAction']
# #if rwdList is changed here, dartEnv2bot.DartEnv2Bot.rwdCompsUsed will also be changed; to reset will require reloading dart_env_2bot.py
#binary digits to use to represent given list of desired rewards
rwdDigsToUse = dartEnv2bot.DartEnv2Bot.getRwdsToUseBin(rwdList) 

#if resuming set these to true and to appropriate prefix and dir name to find pkl file : + '/local/' +resumeExpDict['prfx']+'/'+resumeExpDict['dir']+'/params.pkl'
resumeExp=False 
#exp-011101110111_bl_64-64_pl_128-64-64_it_3835_mxPth_500_nBtch_200000_ALG_TRPO__2018_09_10_13_32_21_0001
resumExpDict={'prfx':'--put prefix here--', 'dir':'-- put dir where old run of experiment can be found, within rllab/local here --'}          
#example :
#resumExpDict={'prfx':'exp-100000000000-bl-64-64-pl-128-64-32-it-1000-mxPth-1000-nBtch-200000-ALG-TRPO-','dir': 'exp-100000000000_bl_64-64_pl_128-64-32_it_1000_mxPth_1000_nBtch_200000_ALG_TRPO__2018_09_03_09_59_00_0001'}
#resumExpDict={'prfx':'exp-011101110111-bl-64-64-pl-128-64-64-it-5000-mxPth-500-nBtch-200000-ALG-TRPO-','dir': 'exp-011101110111_bl_64-64_pl_128-64-64_it_3835_mxPth_500_nBtch_200000_ALG_TRPO__2018_09_10_13_32_21_0001'}

#force use training mode in environment - this disables viewer
verifyTrain = dartEnv2bot.DartEnv2Bot.setTrainState(True)
print('Training is set to : {}'.format(verifyTrain))

#build experiment dict - this is always used to train policy (changes static variables in dart_env_2bot to do this)
expLiteDict = tFuncs.buildExpLiteDict(rwdDigsToUse, envName, itrs=itersList[i], blArch=blArchList[bl], polArch=polArchList[pl], numBatches=numBatches, 
                                    maxPathLength=maxPathLength, useCG=useCG, gae_lambda=gae_lambda,optimizerArgs=optimizerArgs, 
                                    resumeExp=resumeExp, isNorm=isNorm, resumeExpDict=resumExpDict, retrain=False) 

#check appropriate setting of clammping A  :for now use (not isNorm) - if normed then don't clamp, if not normed then do
verifyClampA = dartEnv2bot.DartEnv2Bot.setClampAction(not isNorm)
print('Clamping action before deriving Tau : {}'.format(verifyClampA))

run_experiment_lite(
    run_task,
    # Number of parallel workers for samplingexi
    n_parallel=10,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    #prefix for saving results - directory name where experiment is saved
    exp_prefix=expLiteDict['expPrefix'],
    #resume training - comment out if not using because format has changed
    resume_from=expLiteDict['resumeData'],
    #feed arguments to run_task via variant -> variant= <specified dict of args>, like :
    variant=expLiteDict,
    #set to true for GPU -needs cudnn and CNMeM, currenlty both are disabled/NA
    #use_gpu=True,
    # Specifies the seed for the experiment. If this is not provided, a random seed will be used
    seed=seeds[sd]#1099,  #99,
    #plot=True
)

_,expNameStr, _ = tFuncs.expDirNameToString(expLiteDict['expPrefix'])
print('Finished Training policy :\n{}'.format(expNameStr))