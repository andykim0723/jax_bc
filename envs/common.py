import gym
import d4rl
import numpy as np
from jaxbc.utils.jaxbc_utils import yielding

from RLBench.rlbench.action_modes.action_mode import MoveArmThenGripper
from RLBench.rlbench.action_modes.arm_action_modes import JointVelocity
from RLBench.rlbench.action_modes.gripper_action_modes import Discrete
from RLBench.rlbench.environment import Environment
from RLBench.rlbench.tasks import ReachTarget

def set_env(cfg):
    if cfg['env_name'] == "d4rl":
        env = gym.make(cfg['task_name'])
        return env

    elif cfg['env_name'] == "rlbench":
        import RLBench.rlbench.gym
        obs_type = "vision"
        env = gym.make(cfg['task_name'] + '-' + obs_type + '-v0')
        return env
        # action_mode = MoveArmThenGripper(
        # arm_action_mode=JointVelocity(),
        # gripper_action_mode=Discrete()
        # )
        # env = Environment(action_mode)
        # env.launch()
        # task = env.get_task(ReachTarget)
        # print(env.action_shape)
        # print(env.ob)

        # return task, env