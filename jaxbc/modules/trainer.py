import os
import wandb
import numpy as np
from typing import Dict
from datetime import datetime

from jaxbc.modules.low_policy.low_policy import MLPpolicy
from envs.eval_func import d4rl_evaluate

class OnlineBCTrainer():
    pass
    # raise NotImplementedError("not yet implemented")


class BCTrainer():
    # def __init__
    def __init__(
        self,
        cfg: Dict,
    ):
        self.cfg = cfg
        self.batch_size = cfg['train']['batch_size']

        # string to model
        if cfg['policy']["low_policy"] == "bc":
            self.low_policy = MLPpolicy(cfg=cfg)

        self.n_update = 0
        self.eval_rewards = []
        
        self.log_interval = cfg['interval']['log_interval']
        self.save_interval = cfg['interval']['save_interval']
        self.eval_interval = cfg['interval']['eval_interval']
        self.weights_path = cfg['train']['weights_path']
        self.wandb_record =  cfg['wandb']['record']
        self.eval_env = cfg['eval']['env']
        self.prepare_run()

    def run(self,replay_buffer,env):  
        #
        for step in range(int(self.cfg['train']['steps'])):
            replay_data = replay_buffer.sample(batch_size = self.batch_size)
            info = self.low_policy.update(replay_data)
            self.n_update += 1

            if (self.n_update % self.log_interval) == 0:
                self.print_log(self.n_update,info)
        
            if (self.n_update % self.save_interval) == 0:
                self.save(str(self.n_update)+"_")
            
            if (self.n_update % self.eval_interval) == 0:
                reward_mean = np.mean(self.evaluate(env))
                self.eval_rewards.append(reward_mean)

                print(f"🤯eval🤯 timestep: {self.n_update} | reward mean : {reward_mean}")

                if max(self.eval_rewards) == reward_mean:
                    self.save('best')

                if self.wandb_record:
                    self.wandb_logger.log({
                        "evaluation reward": reward_mean
                    })

            if self.wandb_record:
                self.record(info)
    
    def evaluate(self,env):
        num_episodes = self.cfg['eval']['num_episodes']

        if self.eval_env == "d4rl":
            rewards = d4rl_evaluate(env,self.low_policy,num_episodes)

        return rewards

    def save(self,path):
        save_path = os.path.join(self.weights_path,path)
        self.low_policy.save(save_path)

    def record(self,info):
        loss = info['decoder/mse_loss']

        if self.wandb_record:
            self.wandb_logger.log({
                "mse loss": loss
                }
            )

    def print_log(self,step,info):
        now = datetime.now()
        elapsed = (now - self.start).seconds
        loss = info['decoder/mse_loss']

        print(f"🤯train🤯 timestep: {step} | mse loss : {loss} | elapsed: {elapsed}s")

    def prepare_run(self):
        self.start = datetime.now()

        if self.wandb_record:
            self.wandb_logger = wandb.init(
                project=self.cfg["wandb"]["project"],
                entity=self.cfg["wandb"]["entity"],
                config=self.cfg,
                name=self.cfg["wandb"]["name"]
            )


