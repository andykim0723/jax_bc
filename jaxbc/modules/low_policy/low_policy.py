import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax.linen.module import init

from jaxbc.modules.common import Model
from jaxbc.modules.updates import bc_mlp_updt
from jaxbc.modules.architecture.resnet18_mlp import PrimRN18MLP,PrimMLP
from jaxbc.modules.forwards import resnet18_mlp_forward as fwd


class MLPpolicy():
    
    # initial variables
    def __init__(
            self,
            cfg: Dict,
            init_build_model: bool = True
    ):
        self.observation_dim = cfg['policy_args']['observation_dim']
        self.seed = cfg['seed']
        self.cfg = cfg

        self.rng = jax.random.PRNGKey(self.seed)

        self.__model = None

        if init_build_model:
            self.build_model()

    @property
    def model(self) -> Model:
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value      

    def build_model(self):

        act_scale = False
        action_dim = self.cfg['policy_args']['action_dim']
        # net_arch = [256]*4
        net_arch = self.cfg['policy_args']['architecture']
        activation_fn = nn.relu
        dropout = 0.0
        squash_output = self.cfg['policy_args']['tanh_action']
        layer_norm = False

        if self.cfg['policy_args']['feature_extractor']:
            if self.cfg['policy_args']['feature_extractor'] == 'resnet18':
                mlp = PrimRN18MLP(
                    act_scale=act_scale,
                    output_dim=action_dim,
                    net_arch=net_arch,
                    activation_fn=activation_fn,
                    dropout=dropout,
                    squash_output=squash_output,
                    layer_norm=layer_norm
                )
        else:
            mlp = PrimMLP(
                act_scale=act_scale,
                output_dim=action_dim,
                net_arch=net_arch,
                activation_fn=activation_fn,
                dropout=dropout,
                squash_output=squash_output,
                layer_norm=layer_norm
            )            
        
        init_obs = np.expand_dims(np.zeros((self.observation_dim)),axis=0)

        rng, param_key, dropout_key, batch_key = jax.random.split(self.rng, 4)

        self.rng = rng
        rngs = {"params": param_key, "dropout": dropout_key, "batch_stats": batch_key}
        tx = optax.adam(self.cfg['info']["lr"])
        self.model = Model.create(model_def=mlp, inputs=[rngs, init_obs], tx=tx)

    def update(self, replay_data):
        
        obs = []
        actions = []
        maskings = []

        for data in replay_data:
            obs.append(data['obs'])
            actions.append(data['actions'])
        
        obs = np.concatenate(obs)
        actions = np.concatenate(actions)

        # TODO
        maskings = None

        new_model, info = bc_mlp_updt(
			rng=self.rng,
			mlp=self.model,
			observations=obs,
			actions=actions,
			maskings=maskings
		)  
 
        self.model = new_model
        self.rng, _ = jax.random.split(self.rng)

        return info

    def predict(
        self,
        observations: np.ndarray,
        to_np: bool = True,
        squeeze: bool = False,
        *args, **kwargs  # Do not remove these dummy parameters.
    ) -> np.ndarray:

        self.rng, prediction = fwd(
            rng=self.rng,
            model=self.model,
            observations=observations,
        )

        if squeeze:
            prediction = np.squeeze(prediction, axis=0)

        if to_np:
            return np.array(prediction)
        else:
            return prediction

    def evaluate(
        self,
        replay_data
    ) -> Dict:
        observations = replay_data.observations
        actions = replay_data.actions[:, -1, ...]
        if self.cfg["use_optimal_lang"]:
            raise NotImplementedError("Obsolete")
        maskings = replay_data.maskings[:, -1]

        if maskings is None:
            raise NotImplementedError("No mask")
        maskings = maskings.reshape(-1, 1)
        
        pred_actions = self.predict(observations=observations)

        pred_actions = pred_actions.reshape(-1, self.action_dim) * maskings
        target_actions = actions.reshape(-1, self.action_dim) * maskings
        mse_error = np.sum(np.mean((pred_actions - target_actions) ** 2, axis=-1)) / np.sum(maskings)
        eval_info = {
            "decoder/mse_error": mse_error,
            "decoder/mse_error_scaled(x100)": mse_error * 100
        }
        return eval_info

    def save(self, path: str) -> None:
        self.model.save_dict_from_path(path)

    def load(self, path: str) -> None:
        self.model = self.model.load_dict_from_path(path)
