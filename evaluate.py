import os
import gym
import d4rl # need this for gym env creation
import json
import argparse
import numpy as np

from jaxbc.modules.low_policy.low_policy import MLPpolicy
from envs.eval_func import d4rl_evaluate

def main(args):
    ### cfg ###
    config_filepath = os.path.join('configs',args.mode+'.json')
    with open(config_filepath) as f:
        cfg = json.load(f)

    ### env ###
    env = gym.make(cfg['env'])
    cfg['observation_dim'] = env.observation_space.shape
    cfg['action_dim'] = int(np.prod(env.action_space.shape))

    ### policy ###
    low_policy = MLPpolicy(cfg=cfg)
    load_path = os.path.join('logs',args.load_path)
    low_policy.load(load_path)

    ### evaluation ###
    num_episodes = cfg['eval']['eval_episodes']
    reward_mean = np.mean(d4rl_evaluate(env,low_policy,num_episodes))
    print("rewards: ", reward_mean))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",type=str, default="halfcheetah_bc",
        choices=['halfcheetah_bc','hopper_bc'])
    
    parser.add_argument(
        "--load_path",type=str, default="weights/hopper_bc/best")
    
    args = parser.parse_args()
    main(args)



