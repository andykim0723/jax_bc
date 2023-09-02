import os
import json
import argparse

import gym
import d4rl

import numpy as np
import pickle as pkl

from jaxbc.modules.trainer import BCTrainer,OnlineBCTrainer
from jaxbc.buffers.d4rlbuffer import d4rlBuffer
from jaxbc.buffers.rlbenchbuffer import RlbenchStateBuffer
from jaxbc.utils.jaxbc_utils import yielding

from envs.common import set_env
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
    env = set_env(cfg)
   
    if cfg['env_name'] == "d4rl":
        # env = gym.make(cfg['task_name'])
        cfg['observation_dim'] = env.observation_space.shape
        cfg['action_dim'] = int(np.prod(env.action_space.shape))

        episodes = d4rl.sequence_dataset(env)
        episodes = list(yielding(episodes))
        # observation -> type: numpy.ndarray, shape: (timestep,state)

        replay_buffer = d4rlBuffer(cfg,env=env)
        replay_buffer.add_episodes_from_d4rl(episodes)

    elif cfg['env_name'] == "rlbench":

        # cfg['observation_dim'] = 512 + int(np.prod(env.observation_space['state'].shape)) # visualk feature size + state
        # cfg['observation_dim'] = int(np.prod(env.observation_space['state'].shape)) 
        # cfg['action_dim'] = int(np.prod(env.action_space.shape))


        # data loading
        print("loading data..")
        data_path = cfg['info']['data_path'] + '/variation0'
        # load task_name 
        with open(data_path+'/variation_descriptions.pkl','rb') as f:
            data = pkl.load(f)
            task_name = data[0]
        print(task_name)
        
        # load data

        episodes = []
        for filename in os.listdir(data_path+"/episodes"):
            episode_path = os.path.join(data_path+"/episodes", filename)
            episode = {}
            with open(episode_path+'/low_dim_obs.pkl','rb') as f:
                data = pkl.load(f)._observations
            # observations = np.concatenate([obs.task_low_dim_state[np.newaxis,:] for obs in data],axis=0)
            observations = np.concatenate([obs.get_low_dim_data()[np.newaxis,:] for obs in data],axis=0)
            actions = np.concatenate([np.append(obs.joint_velocities,[obs.gripper_open])[np.newaxis,:] for obs in data],axis=0)
    
            episode['observations'] = observations
            episode['actions'] = actions

            episodes.append(episode)            


        replay_buffer = RlbenchStateBuffer(cfg,env=env)
        replay_buffer.add_episodes_from_rlbench(episodes)

    trainer = BCTrainer(cfg=cfg)


    
    # train
    trainer.run(replay_buffer,env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Main Args
    
    parser.add_argument(
        "--task",type=str, default="d4rl-halfcheetah",
        choices=['d4rl-halfcheetah','d4rl-hopper','rlbench-pick_and_lift'],
        required=True)

    parser.add_argument(
        "--policy",type=str, default="bc",
        choices=['bc'],
        required=True)
    
    args = parser.parse_args()
    main(args)


