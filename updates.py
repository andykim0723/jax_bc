from typing import Tuple, Dict, Any

import jax
import flax
from jax import numpy as jnp

from andykim_jax.common import Model

Params = flax.core.FrozenDict[str, Any]

@jax.jit
def bc_mlp_updt(
	rng: jnp.ndarray,
	mlp: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	maskings: jnp.ndarray
):
	rng, dropout_key = jax.random.split(rng)
	action_dim = actions.shape[-1]
	if maskings is None:
		maskings = jnp.ones(actions.shape[0])

	observations = observations.squeeze(axis=1)

	# TODO convert obs to feature using cnn
	actions = actions.reshape(-1, action_dim)
	maskings = maskings.reshape(-1, 1)
	# jax.debug.print("🤯 {x} 🤯", x=maskings)
	target_actions = actions * maskings	

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:

		pred_actions = mlp.apply_fn(
			{"params": params},
			observations=observations,
			rngs={"dropout": dropout_key},
			deterministic=False,
			training=True,
		)
		jax.debug.print("target_action: {x} 🤯", x=target_actions)
		pred_actions = pred_actions.reshape(-1, action_dim) * maskings
		jax.debug.print("pred_action: {x} 🤯", x=pred_actions)
		mse_loss = jnp.sum(jnp.mean((pred_actions - target_actions) ** 2, axis=-1)) / jnp.sum(maskings)

		_infos = {
			"decoder/mse_loss": mse_loss,
			"__decoder/pred_actions": pred_actions,
			"__decoder/target_actions": target_actions
		}
		return mse_loss, _infos
	new_mlp, infos = mlp.apply_gradient(loss_fn)

	return new_mlp, infos
