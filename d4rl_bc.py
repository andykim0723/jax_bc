
import random
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

def main():
    
    ### env ###
    env = gym.make('hopper-medium-expert-v0')

    episodes = d4rl.sequence_dataset(env)

    observation_dim = env.observation_space.shape
    action_dim = int(np.prod(env.action_space.shape))


    cfg = {
        'max_iter': 500000//71,
        'batch_size': 32,
        'observation_dim': 11,
        'seed': 42,
        'lr': 1e-4,
        'buffer_size': 1e10,
        'subseq_len': 1,
        'low_policy': 'mlp_policy',
        'architecture': 'naive_mlp',
        'observation_dim': observation_dim,
        'action_dim': action_dim,
        'log_interval': 10000,
        'save_interval': 1e12,
        'save_path': 'logs/'
    }

    trainer = BCTrainer(cfg=cfg)

    # generator to list
    episodes = list(yielding(episodes))
    
    # train
    for n_iter in range(cfg['max_iter']):

        episodes_for_train = deepcopy(episodes)
        random.shuffle(episodes_for_train)

        replay_buffer = BCBuffer(buffer_size=cfg['buffer_size'],subseq_len=cfg['subseq_len'],env=env)
        replay_buffer.add_episodes_from_d4rl(episodes_for_train)
        trainer.run(replay_buffer)

    # eval
    num_episodes = 100
    rewards = []
    for n in range(num_episodes):
        
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

    print("rewards: ", np.mean(rewards))

if __name__ == "__main__":
    main()