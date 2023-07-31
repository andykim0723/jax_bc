import os
import json
import argparse

import gym
import d4rl

import numpy as np

from jaxbc.modules.trainer import BCTrainer,OnlineBCTrainer
from jaxbc.buffers.buffer import BCBuffer
from jaxbc.utils.jaxbc_utils import yielding

def main(args):
 
    ### train info ###
    env_name, task_name = args.task.split('-')
    policy_name = args.policy
    print(f"train info -> task: {task_name} | policy: {policy_name} ")
    
    ### config file ###
    json_fname = task_name + '_' + policy_name
    config_filepath = os.path.join('configs',env_name,json_fname+'.json')
    with open(config_filepath) as f:
        cfg = json.load(f)
    
    ### env ###
    if cfg['env_name'] == "d4rl":
        env = gym.make(cfg['task_name'])
        cfg['observation_dim'] = env.observation_space.shape
        cfg['action_dim'] = int(np.prod(env.action_space.shape))

        episodes = d4rl.sequence_dataset(env)
        episodes = list(yielding(episodes))

    trainer = BCTrainer(cfg=cfg)

    # new 
    replay_buffer = BCBuffer(buffer_size=cfg['info']['buffer_size'],subseq_len=cfg['info']['subseq_len'],env=env)
    replay_buffer.add_episodes_from_d4rl(episodes)
    
    # train
    trainer.run(replay_buffer,env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Main Args
    
    parser.add_argument(
        "--task",type=str, default="d4rl-halfcheetah",
        choices=['d4rl-halfcheetah','d4rl-hopper'],
        required=True)

    parser.add_argument(
        "--policy",type=str, default="bc",
        choices=['bc'],
        required=True)
    
    args = parser.parse_args()
    main(args)



