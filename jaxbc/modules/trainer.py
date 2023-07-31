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
        self.batch_size = cfg['info']['batch_size']

        # string to model
        if cfg["policy"] == "bc":
            self.low_policy = MLPpolicy(cfg=cfg)

        self.n_update = 0
        self.eval_rewards = []

        self.train_steps = cfg['info']['train_steps']
        self.eval_episodes = self.cfg['info']['eval_episodes']
        self.eval_env = cfg['env_name']

        self.log_interval = cfg['info']['log_interval']
        self.save_interval = cfg['info']['save_interval']
        self.eval_interval = cfg['info']['eval_interval']
        self.weights_path = cfg['info']['weights_path']

        self.wandb_record =  cfg['wandb']['record']

        self.prepare_run()

    def run(self,replay_buffer,env):  
        #
        for _ in range(int(self.train_steps)):
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

        if self.eval_env == "d4rl":
            rewards = d4rl_evaluate(env,self.low_policy,self.eval_episodes)

        return rewards

    def save(self,path):
        # date_time = datetime.now().strftime('%m-%d_%H:%M')
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
        
        env_name = self.cfg['env_name']
        task_name = self.cfg['task_name'].split('-')[0]  
        policy_name = self.cfg['policy']

        project = env_name+ '_' + task_name  
        name = task_name + '_' + policy_name
        if self.wandb_record:
            self.wandb_logger = wandb.init(
                project=project,
                name=name,
                entity=self.cfg["wandb"]["entity"],
                config=self.cfg,
            )


