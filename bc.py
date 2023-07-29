import os
import json
import random
import argparse

from copy import deepcopy
from PIL import Image
import jax
import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget, PickAndLift


from jaxbc.modules.trainer import BCTrainer,OnlineBCTrainer
from jaxbc.buffers.buffer import BCBuffer

import gym
import d4rl

class Agent(object):  
    def __init__(self, action_shape):
        self.action_shape = action_shape

    def ingest(self, demos):
        pass

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)

def create_rlbench_env(task):
    ### environment ###
    # To use 'saved' demos, set the path below, and set live_demos=False
    live_demos = True
    DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
    env = Environment(
        action_mode, DATASET, obs_config, False)
    return env

    # task_env = env.get_task(task)
    # # demos = task_env.get_demos(2, live_demos=live_demos)
    # # agent = Agent(env.action_shape)
    # # agent.ingest(demos)
    # return task_env

def yielding(ls):
    for i in ls:
        yield i

def evaluate(env,trainer,n_episode):
    # eval
    rewards = []
    for n in range(n_episode):
        
        obs = env.reset()
        returns = 0
        for t in range(env._max_episode_steps):

            # img_arr = env.render(mode="rgb_array")
            # img = Image.fromarray(img_arr)
            # img.save('test.png')

            action = trainer.low_policy.predict(obs)
            obs,rew,done,info = env.step(action)
            returns += rew
            if done:
                break
        rewards.append(returns)

    return rewards


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
        choices=['halfcheetah_bc'],
        required=True)
    
    # TODO fix later
    # parser.add_argument(
    #     "--env", type=str, default='halfcheetah-expert-v0', 
    #     choices=['halfcheetah-expert-v0'],
    #     help='currently supporting halfcheetah-expert only.'
    #     'will add Pick-Place(RLBench), Carla_route1(CARLA).')
    
    # parser.add_argument(
    #     "--policy", type=str, default='bc', 
    #     choices=['bc'])
    
    # parser.add_argument(
    #     "--feature_extractor", type=str, default=None, 
    #     choices=['resnet18'])
    
    args = parser.parse_args()
    main(args)



