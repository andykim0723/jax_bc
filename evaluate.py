import os
import gym
import d4rl # need this for gym env creation
import json
import argparse
import numpy as np
from jaxbc.modules.low_policy.low_policy import MLPpolicy

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
    load_path = os.path.join('logs','test')
    low_policy.load(load_path)

    ### evaluation ###
    rewards = []
    for _ in range(cfg['eval']['eval_episodes']):

        obs = env.reset()
        returns = 0
        for t in range(env._max_episode_steps):
            action = low_policy.predict(obs)
            obs,rew,done,info = env.step(action)
            returns += rew
            if done:
                break
        rewards.append(returns)
    print("rewards: ", np.mean(rewards))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Main Args
    parser.add_argument(
        "--mode",type=str, default="halfcheetah_bc",
        choices=['halfcheetah_bc','hopper_bc'])
    args = parser.parse_args()
    main(args)



