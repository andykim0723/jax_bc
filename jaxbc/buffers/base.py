from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
# from stable_baselines3.common.vec_env import VecNormalize


try:
	# Check memory used by replay buffer when possible
	import psutil
except ImportError:
	psutil = None

class Episode:
    	
	def __init__(self):
		self.observations = []
		self.next_observations = []
		self.actions = []
		self.rewards = []
		self.dones = []
		self.infos = []
		self.maskings = []
		self.rtgs = []

	def __len__(self):
		return len(self.observations)

	def __getitem__(self, idx: slice):
		if idx.start < 0:
			idx = slice(0, idx.stop, None)
		observations = list(self.observations[idx].copy())
		next_observations = list(self.next_observations[idx].copy())
		actions = list(self.actions[idx].copy())
		rewards = list(self.rewards[idx].copy())
		dones = list(self.dones[idx].copy())
		infos = list(self.infos[idx].copy())
		maskings = list(self.maskings[idx].copy())
		rtgs = list(self.rtgs[idx].copy())
		return Episode.from_list(observations, next_observations, actions, rewards, dones, infos, maskings, rtgs)
    
	@staticmethod
	def from_list(
		observations: List,
		next_observations: List,
		actions: List,
		rewards: List,
		dones: List,
		infos: List,
		maskings: List,
		rtgs: List
	) -> "Episode":
		ret = Episode()
		ret.observations = observations
		ret.next_observations = next_observations
		ret.actions = actions
		ret.rewards = rewards
		ret.dones = dones
		ret.infos = infos
		ret.maskings = maskings
		ret.rtgs = rtgs
		return ret
	
	def set_zeropaddings(self, n_padding: int):
		for i in range(n_padding):
			self.observations.append(np.zeros(self.observation_dim, ) + 1)
			self.next_observations.append(np.zeros(self.observation_dim, ) + 1)
			self.actions.append(np.zeros(self.action_dim, ) + 1)
			self.rewards.append(np.array(0))
			self.dones.append(np.array(True))
			self.infos.append([])
			self.maskings.append(np.array(0))
			self.rtgs.append(0)

class BaseBuffer(ABC):
	"""
	Base class that represent a buffer (rollout or replay)

	:param buffer_size: Max number of element in the buffer
	:param observation_space: Observation space
	:param action_space: Action space
	:param n_envs: Number of parallel environments
	"""

	def __init__(
		self,
		buffer_size: int,
		env,
		n_envs: int = 1
	):
		super(BaseBuffer, self).__init__()
		observation_space = env.observation_space
		action_space = env.action_space

		self.env = env
		self.buffer_size = buffer_size
		self.observation_space = observation_space
		self.action_space = action_space

		self.pos = 0
		self.full = False
		self.n_envs = n_envs

	def __len__(self):
		raise NotImplementedError()

	@staticmethod
	def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
		"""
		Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
		to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
		to [n_steps * n_envs, ...] (which maintain the order)

		:param arr:
		:return:
		"""
		shape = arr.shape
		if len(shape) < 3:
			shape = shape + (1,)
		return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

	def size(self) -> int:
		"""
		:return: The current size of the buffer
		"""
		if self.full:
			return self.buffer_size
		return self.pos

	def add(self, *args, **kwargs) -> Tuple[int, int]:
		"""
		Add elements to the buffer.
		"""
		raise NotImplementedError()

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
		env,
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

	@abstractmethod
	def _get_samples(
		self,
		batch_inds: np.ndarray,
		env_inds: np.ndarray,
		env,
		get_batch_inds: bool = False
	):
		raise NotImplementedError()

	@staticmethod
	def _normalize_obs(
		obs: Union[np.ndarray, Dict[str, np.ndarray]],
		env,
	) -> Union[np.ndarray, Dict[str, np.ndarray]]:
		if env is not None:
			return env.normalize_obs(obs)
		return obs

	@staticmethod
	def _normalize_reward(reward: np.ndarray, env = None) -> np.ndarray:
		if env is not None:
			return env.normalize_reward(reward).astype(np.float32)
		return reward
