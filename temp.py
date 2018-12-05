# -*- coding: utf-8 -*-
import os
#trpo 
import rllab.misc.logger as logger
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
#from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
#from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

#set this to only save snapshots

logger.set_snapshot_dir('snapShots16_8')
logger.set_snapshot_mode('last')
#logger.set_snapshot_gap(100)

env = normalize(GymEnv('DartWalker3d-v1'))
env.render()

policy16 = GaussianMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(16,8)
    #hidden_sizes=(128,128,64)
    #hidden_sizes=(64, 32, 16)
    #hidden_sizes=(32, 32, 16)
)
baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy16,
    baseline=baseline,
    batch_size=4000,
    whole_paths=True,
    max_path_length=1000,
    n_itr=2000,
    discount=0.99,
    step_size=0.01,
    start_itr=0
)
algo.train()

#build and set snapshot directory based on policy vars, build policy
#polValTpl is tuple of policy architecture (hidden layer count)
def buildPolicy(polValTpl, env):
    #directory to save results for this run - built off policy architecture
    snapShotDir = 'snapShots' + '_'.join([str(x) for x in polValTpl])
    print('snapShotDir : ' + snapShotDir)
    
    if not os.path.exists(snapShotDir):
        os.makedirs(snapShotDir)
    
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=polValTpl
    )
    
    return policy, snapShotDir

#train algorithm for TRPO using passed policy and environment
def trainPolicy(policy, env, snapShotDir):
        #set logger to save last result to snapShotDir
    logger.set_snapshot_dir(snapShotDir)
    logger.set_snapshot_mode('last')
    
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        whole_paths=True,
        max_path_length=1000,
        n_itr=2000,
        discount=0.99,
        step_size=0.01,
        start_itr=0
    )
    algo.train()
    
    return algo
    
    

usePolicy(policy16, env)
usePolicy(policy, env)
usePolicy(policy, env)

#consume trained policy to see how it performs in env
def usePolicy(policy, env):
    #consume policy
    observations = []
    actions = []
    rewards = []
    
    observation = env.reset()
    env.render()
    iter = 0
    for i in range(30000):
    #while i < 30000:
        # policy.get_action() returns a pair of values. The second one returns a dictionary, whose values contain
        # sufficient statistics for the action distribution. It should at least contain entries that would be
        # returned by calling policy.dist_info(), which is the non-symbolic analog of policy.dist_info_sym().
        # Storing these statistics is useful, e.g., when forming importance sampling ratios. In our case it is
        # not needed.
        action, _ = policy.get_action(observation)
        # Recall that the last entry of the tuple stores diagnostic information about the environment. In our
        # case it is not needed.
        next_observation, reward, terminal, _ = env.step(action)
        env.render()
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        observation = next_observation
        iter = iter+1
        if terminal:
            # Finish rollout if terminal state reached
            print('Terminal break after %s iters' %(str(iter)))
            observations = []
            actions = []
            rewards = []
            
            observation = env.reset()
            env.render()
            iter = 0
            break
    
#terminate passed environment
def killEnv(env):
    env.terminate()
