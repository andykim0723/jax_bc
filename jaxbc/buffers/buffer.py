import random
from copy import deepcopy
from typing import Dict, Optional, Union, Tuple, List

import h5py
import numpy as np
from jax.tree_util import tree_map
# from stable_baselines3.common.vec_env import VecNormalize

from jaxbc.buffers.base import BaseBuffer
# from comde.rl.buffers.buffers.episodic import EpisodicMaskingBuffer
# from comde.rl.buffers.episodes.source_target_skill import SourceTargetSkillContainedEpisode
# from comde.rl.buffers.episodes.source_target_state import SourceStateEpisode
# from comde.rl.buffers.type_aliases import ComDeBufferSample
# from comde.rl.envs.base import ComdeSkillEnv
# from comde.rl.envs.utils.skill_to_vec import SkillInfoEnv
# from comde.utils.common.misc import get_params_for_skills

### data format ###
# data = {
#   "observations": ...,
#    "actions": ...,
#    "next_observations": ..., 
# }


class BCBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        subseq_len: int,
        env = None, 
        n_envs: int = 1
        ):
        # observation_space = env.observation_space
        # action_space = env.action_space
        observation_space = None
        action_space = None
        self.env = env
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
		# self.obs_shape = get_obs_shape(observation_space)
		# self.action_dim = get_action_dim(action_space)

        self.observation_dim = (1,224,224,3) 
        self.action_dim = (1,2)

        self.pos = 0
        self.full = False
        self.n_envs = n_envs

        self.subseq_len = subseq_len
        self.episodes = []  # type: List[Episode]
        self.episode_lengths = []

    def __len__(self):
        return len(self.episodes)

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, episode) -> None:

        assert len(episode['obs']) > self.subseq_len, \
            "Too short episode. Please remove this episode or decrease the subseq len."
        self.episodes.append(episode)
        self.pos += 1
    
    def add_episodes_from_h5py(
        self,
        episodes
    ):
        for episode in episodes:
            self.add(episode) 
            ep_len = len(episode['obs'])
            self.episode_lengths.append(ep_len)


        if len(self.episode_lengths) == 0:
            return False

        self.min_episode_length = min(self.episode_lengths)
        self.max_episode_length = max(self.episode_lengths)
        
        ## zero padding for subsequence

        n_padding = self.subseq_len
        for ep in self.episodes:
            for i in range(n_padding):
                ep['obs'].append(np.zeros(self.observation_dim, ) + 1)
                ep['actions'].append(np.zeros(self.action_dim, ) + 1)
                ep['next_obs'].append(np.zeros(self.observation_dim, ) + 1)    

            # ep.set_zeropaddings(n_padding=self.subseq_len)
        return True       
        
    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.   
        """
        self.pos = 0
        self.full = False

    def sample(
        self,
        env = None,
        batch_size: int = None,
        batch_inds: np.ndarray = None,
        get_batch_inds: bool = False
    ):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :param batch_inds
        :param get_batch_inds
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos

        if batch_inds is None:
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
 
        env_inds = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))


        return self._get_samples(batch_inds, env_inds=env_inds, env=env, get_batch_inds=get_batch_inds)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_inds: np.ndarray,
        env,
        get_batch_inds: bool = False
    ):

        subtrajectories = []
        timesteps = []
        
        episodes = [self.episodes[batch_idx] for batch_idx in batch_inds]

        for episode in episodes:
            ep_len = len(episode['obs'])
                # for starting_idx in starting_idxs:
            starting_idx = np.random.randint(0, ep_len - self.subseq_len)

            subtrajectory = {}
            for k,v in episode.items():
                subtrajectory[k] = v[starting_idx: starting_idx + self.subseq_len]
            
            # TODO
            subtrajectory['maskings'] = None
            subtrajectories.append(subtrajectory)
            timesteps.append(np.arange(starting_idx, starting_idx + self.subseq_len))

      
        return subtrajectories

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs
    
