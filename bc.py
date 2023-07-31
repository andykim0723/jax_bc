import os
import json
import argparse

import gym
import d4rl

import jax
import numpy as np

from jaxbc.modules.trainer import BCTrainer,OnlineBCTrainer
from jaxbc.buffers.buffer import BCBuffer
from jaxbc.utils.jaxbc_utils import yielding

def main(args):

    config_filepath = os.path.join('configs',args.mode+'.json')

    with open(config_filepath) as f:
        cfg = json.load(f)
    
    ### env ###
    env = gym.make(cfg['env'])
    episodes = d4rl.sequence_dataset(env)
    cfg['observation_dim'] = env.observation_space.shape
    cfg['action_dim'] = int(np.prod(env.action_space.shape))

    trainer = BCTrainer(cfg=cfg)

    # generator to list
    episodes = list(yielding(episodes))

    # new 
    replay_buffer = BCBuffer(buffer_size=cfg['train']['buffer_size'],subseq_len=cfg['train']['subseq_len'],env=env)
    replay_buffer.add_episodes_from_d4rl(episodes)
    
    # train
    trainer.run(replay_buffer,env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Main Args
    
    parser.add_argument(
        "--mode",type=str, default="halfcheetah_bc",
        choices=['halfcheetah_bc','hopper_bc'],
        required=True)
    
    args = parser.parse_args()
    main(args)



