import os
import wandb
import numpy as np
from typing import Dict
from datetime import datetime

from jaxbc.modules.low_policy.low_policy import MLPpolicy


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
        
        self.log_interval = cfg['train']['log_interval']
        self.save_interval = cfg['train']['save_interval']
        self.eval_interval = cfg['eval']['eval_interval']
        self.weights_path = cfg['train']['weights_path']
        self.wandb_record =  cfg['wandb']['record']
        self.prepare_run()


    def run(self,replay_buffer,env):  
        #
        for step in range(int(self.cfg['train']['steps'])):
            replay_data = replay_buffer.sample(batch_size = self.batch_size)
            info = self.low_policy.update(replay_data)
            self.n_update += 1

            if (self.n_update % self.log_interval) == 0:
                print(self.n_update,info['decoder/mse_loss'])
                self.print_log(self.n_update,info)
        
            if (self.n_update % self.save_interval) == 0:
                self.save(str(self.n_update)+"_")
            
            if (self.n_update % self.eval_interval) == 0:
                print(f'🤯start evaluation🤯')
                reward_mean = np.mean(self.evaluate(env))
                print(f"timestep: {self.n_update} | reward mean : {reward_mean}")

                self.eval_rewards.append(reward_mean)
                if max(self.eval_rewards) == reward_mean:
                    self.save('best')

                if self.wandb_record:
                    self.wandb_logger.log({
                        "evaluation reward": reward_mean
                    })

            # log loss
            if self.wandb_record:
                self.record(info)

    def print_log(self,step,info):
        
        now = datetime.now()
        elapsed = (now - self.start).seconds
        loss = info['decoder/mse_loss']

        print(f"timestep: {step} | mse loss : {loss} | elapsed: {elapsed}s")
    
    def record(self,info):
                
        loss = info['decoder/mse_loss']
        if self.wandb_record:
            self.wandb_logger.log({
                "mse loss": loss
                }
            )

    def save(self,path):
        save_path = os.path.join(self.weights_path,path)
        self.low_policy.save(save_path)


    def evaluate(self,env):
        # eval
        rewards = []
        for n in range(self.cfg['eval']['eval_episodes']):

            obs = env.reset()
            returns = 0
            for t in range(env._max_episode_steps):

                # img_arr = env.render(mode="rgb_array")
                # img = Image.fromarray(img_arr)
                # img.save('test.png')

                action = self.low_policy.predict(obs)
                obs,rew,done,info = env.step(action)
                returns += rew
                if done:
                    break
            rewards.append(returns)

        return rewards


    def prepare_run(self):
        self.start = datetime.now()

        if self.wandb_record:
            self.wandb_logger = wandb.init(
                project=self.cfg["wandb"]["project"],
                entity=self.cfg["wandb"]["entity"],
                config=self.cfg,
                name=self.cfg["wandb"]["name"]
            )


