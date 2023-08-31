
import jax

import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget, PickAndLift


from jaxbc.modules.trainer import BCTrainer,OnlineBCTrainer
from jax_bc.jaxbc.buffers.d4rlbuffer import BCBuffer


class Agent(object):  
    def __init__(self, action_shape):
        self.action_shape = action_shape

    def ingest(self, demos):
        pass

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)

def create_env(task):
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

def random_episodes(num_episodes):
    rng = jax.random.PRNGKey(42)

    episodes = []
    num_episodes = num_episodes
    for i_epi in range(num_episodes):
        num_ts =  np.random.randint(low=5,high=100)
        obs = []
        actions = []
        for ts in range(num_ts):
            img_rng, action_rng = jax.random.split(rng, num=2) # not working, sampling same number each loop
            
            img = np.random.rand(224,224,3)
            action = np.random.randint(low=0,high=5,size=2)

            obs.append(np.expand_dims(img,axis=0))
            actions.append(np.expand_dims(action,axis=0))

        

        nxt_obs = obs[1:]
        dummy_img = np.zeros(shape=(1,224,224,3))
        nxt_obs.append(dummy_img)
        
        # np_obs = np.concatenate(obs)
        # np_nxt_obs = np.concatenate(nxt_obs)
        # np_actions = np.concatenate(actions)
        episode = {
            'obs': obs,
            'actions': actions,
            'next_obs': nxt_obs
        }
        episodes.append(episode)
    
    return episodes


def main():
    
    ### env ###
    task = PickAndLift
    env = create_env(task=task)
    env.launch()
    task_env = env.get_task(task)


    # use task_env to step & reset, not env

    online = False

    cfg = {
        'batch_size': 32,
        'observation_dim': 512,
        'seed': 42,
        'lr': 1e-2
    }


    if online:
        trainer = OnlineBCTrainer()
    else:
        trainer = BCTrainer(cfg=cfg)


    buffer_size = 1e10
    subseq_len = 1
    replay_buffer = BCBuffer(buffer_size=buffer_size,subseq_len=subseq_len)
    episodes = random_episodes(num_episodes=100)

    # ep_len = [len(ep['obs']) for ep in episodes]

    replay_buffer.add_episodes_from_h5py(episodes)

    trainer.run(replay_buffer)

    env.shutdown()



if __name__ == "__main__":
    main()