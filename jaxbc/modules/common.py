import os
from typing import Any, Optional, Tuple, Union, Callable, Sequence, Type, TypeVar, List, Callable

T = TypeVar('T')

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


from jaxbc.modules.type_aliases import (
	PRNGKey,
	Shape,
	Dtype,
	Array,
	Params
)
    	


class Scaler(nn.Module):
	"""
		Scaling the output of base model
	"""
	base_model: nn.Module
	scale: jnp.ndarray

	@nn.compact
	def __call__(self, *args, **kwargs):
		original_output = self.base_model(*args, **kwargs)
		ret = self.scale * original_output
		return ret


def create_mlp(
	output_dim: int,
    net_arch: List[int],
    activation_fn: Callable = nn.relu,
    dropout: float = 0.0,
    squash_output: bool = False,
    layer_norm: bool = False,
    batch_norm: bool = False,
    use_bias: bool = True,
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal(),
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
):
	if output_dim > 0:
		net_arch = list(net_arch)
		net_arch.append(output_dim)

	return MLP(
		net_arch=net_arch,
		activation_fn=activation_fn,
		dropout=dropout,
		squash_output=squash_output,
		layer_norm=layer_norm,
		batch_norm=batch_norm,
		use_bias=use_bias,
		kernel_init=kernel_init,
		bias_init=bias_init
	)


class MLP(nn.Module):
	net_arch: List
	activation_fn: nn.Module
	dropout: float = 0.0
	squash_output: bool = False

	layer_norm: bool = False
	batch_norm: bool = False
	use_bias: bool = True
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal()
	bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

	@nn.compact
	def __call__(self, x: jnp.ndarray, deterministic: bool = False, training: bool = True):
		for features in self.net_arch[: -1]:
			x = nn.Dense(features=features, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
			if self.batch_norm:
				x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
			if self.layer_norm:
				x = nn.LayerNorm()(x)

			x = self.activation_fn(x)
			x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)

		if len(self.net_arch) > 0:
			x = nn.Dense(features=self.net_arch[-1], kernel_init=self.kernel_init, bias_init=self.bias_init)(x)

		if self.squash_output:
			return nn.tanh(x)
		else:
			return x



@flax.struct.dataclass
class Model:
	step: int
	apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
	params: Params
	batch_stats: Union[Params]
	tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
	opt_state: Optional[optax.OptState] = None
	# model_cls: Type = None

	@classmethod
	def create(
		cls,
		model_def: nn.Module,
		inputs: Sequence[jnp.ndarray],
		tx: Optional[optax.GradientTransformation] = None,
		**kwargs
	) -> 'Model':

		variables = model_def.init(*inputs)

		_, params = variables.pop('params')

		"""
		NOTE:
			Here we unfreeze the parameter. 
			This is because some optimizer classes in optax must receive a dict, not a frozendict, which is annoying.
			https://github.com/deepmind/optax/issues/160
			And ... if we can access to the params, then why it should be freezed ? 
		"""
		# NOTE : Unfreeze the parameters !!!!!
		params = params.unfreeze()

		# Frozendict's 'pop' method does not support default value. So we use get method instead.
		batch_stats = variables.get("batch_stats", None)

		if tx is not None:
			opt_state = tx.init(params)
		else:
			opt_state = None

		return cls(
			step=1,
			apply_fn=model_def.apply,
			params=params,
			batch_stats=batch_stats,
			tx=tx,
			opt_state=opt_state,
			**kwargs
		)

	def __call__(self, *args, **kwargs):
		return self.apply_fn({"params": self.params}, *args, **kwargs)

	def apply_gradient(
		self,
		loss_fn: Optional[Callable[[Params], Any]] = None,
		grads: Optional[Any] = None,
		has_aux: bool = True
	) -> Union[Tuple['Model', Any], 'Model']:

		assert ((loss_fn is not None) or (grads is not None), 'Either a loss function or grads must be specified.')

		if grads is None:
			grad_fn = jax.grad(loss_fn, has_aux=has_aux)
			if has_aux:
				grads, aux = grad_fn(self.params)
			else:
				grads = grad_fn(self.params)
		else:
			assert (has_aux, 'When grads are provided, expects no aux outputs.')

		updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
		new_params = optax.apply_updates(self.params, updates)
		new_model = self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state)

		if has_aux:
			return new_model, aux
		else:
			return new_model

	def save_dict_from_path(self, save_path: str) -> Params:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		with open(save_path, 'wb') as f:
			f.write(flax.serialization.to_bytes(self.params))
		return self.params

	def load_dict_from_path(self, load_path: str) -> "Model":
		with open(load_path, 'rb') as f:
			params = flax.serialization.from_bytes(self.params, f.read())
		return self.replace(params=params)

	def save_batch_stats_from_path(self, save_path: str) -> Params:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		with open(save_path, 'wb') as f:
			f.write(flax.serialization.to_bytes(self.batch_stats))
		return self.batch_stats

	def load_batch_stats_from_path(self, load_path: str) -> "Model":
		with open(load_path, 'rb') as f:
			batch_stats = flax.serialization.from_bytes(self.batch_stats, f.read())
		return self.replace(batch_stats=batch_stats)

	def load_dict(self, params: bytes) -> 'Model':
		params = flax.serialization.from_bytes(self.params, params)
		return self.replace(params=params)
