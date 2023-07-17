from typing import Dict
# from andykim_jax.model import Model
from jaxbc.modules.low_policy import MLPpolicy

### class trainer ###

class OnlineBCTrainer():
    pass
    # raise NotImplementedError("not yet implemented")


class BCTrainer():
    # def __init__
    def __init__(
        self,
        cfg: Dict,
    ):
        self.batch_size = cfg['batch_size']
        
        #TODO initialize low_policy

        seed = cfg['seed']
        self.low_policy = MLPpolicy(seed=seed,cfg=cfg)
        self.n_update = 0
        self.log_interval = 1e20
        self.save_interval = 2000

    def run(self,replay_buffer):
        for _ in range(len(replay_buffer)//self.batch_size):
            replay_data = replay_buffer.sample(batch_size = self.batch_size)

            info = self.low_policy.update(replay_data)
            self.n_update += 1

        if (self.n_update % self.log_interval) == 0:
                self.record(step=self.n_update)

        if (self.n_update % self.save_interval) == 0:
            self.save()
    
    def record():
        raise NotImplementedError()

    def save(self):
        for key, save_path in self.cfg["save_paths"].items():
            getattr(self, key).save(save_path)   
    
    def evaluate(self,replay_buffer):
        
        eval_data = replay_buffer.sample(128)  
        info = self.low_policy.evaluate(eval_data)

