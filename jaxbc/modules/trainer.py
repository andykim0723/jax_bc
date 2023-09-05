import os
import cv2
import wandb
import numpy as np
from typing import Dict
from datetime import datetime

from jaxbc.modules.low_policy.low_policy import MLPpolicy
from jaxbc.utils.common import save_video
from envs.eval_func import d4rl_evaluate,rlbench_evaluate

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
        task_name = cfg['task_name']
        policy_name = cfg['policy']
        self.save_path= os.path.join(cfg['info']['save_path'],f"{task_name}_{policy_name}")
        if policy_name == "bc":
            self.low_policy = MLPpolicy(cfg=cfg)

        self.n_update = 0
        self.eval_rewards = []
        self.success_rates = []

        self.train_steps = cfg['info']['train_steps']
        self.eval_episodes = self.cfg['info']['eval_episodes']
        self.eval_env = cfg['env_name']

        self.log_interval = cfg['info']['log_interval']
        self.save_interval = cfg['info']['save_interval']
        self.eval_interval = cfg['info']['eval_interval']
        self.weights_path = os.path.join(self.save_path,"weights") 
        self.log_path = os.path.join(self.save_path,"logs") 
        self.video_path = os.path.join(self.save_path,"videos") if cfg['info']['record_video'] else None
        
        os.makedirs(self.weights_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        if self.video_path:
            os.makedirs(self.video_path, exist_ok=True)
    
        self.wandb_record =  cfg['wandb']['record']

        self.prepare_run()

    def run(self,replay_buffer,env):  
        #
        for _ in range(int(self.train_steps)):
            replay_data = replay_buffer.sample(batch_size = self.batch_size)
            # actions = np.squeeze(np.array([rep['actions'] for rep in replay_data]))
            info = self.low_policy.update(replay_data)
            self.n_update += 1

            if (self.n_update % self.log_interval) == 0:
                self.print_log(self.n_update,info)
        
            if (self.n_update % self.save_interval) == 0:
                self.save(str(self.n_update)+"_")
            
            if (self.n_update % self.eval_interval) == 0:
                if self.eval_env == "d4rl":
                    reward_mean = np.mean(self.evaluate(env))
                    self.eval_rewards.append(reward_mean)
                    print(f"🤯eval🤯 timestep: {self.n_update} | reward mean : {reward_mean}")

                    if max(self.eval_rewards) == reward_mean:
                        self.save('best')

                    if self.wandb_record:
                        self.wandb_logger.log({
                            "evaluation reward": reward_mean
                        })
                elif self.eval_env == "rlbench":
                    success_rate = self.evaluate(env)
                    self.success_rates.append(success_rate)
                    print(f"🤯eval🤯 timestep: {self.n_update} | success_rate mean : {success_rate}")

                    if max(self.success_rates) == success_rate:
                        self.save('best')
                    if self.wandb_record:
                        self.wandb_logger.log({
                        "success rates": success_rate
                    })

            if self.wandb_record:
                self.record(info)
    
    def evaluate(self,env):

        if self.eval_env == "d4rl":
            rewards = d4rl_evaluate(env,self.low_policy,self.eval_episodes)
            return rewards
        elif self.eval_env == "rlbench":
            success_rate,frames = rlbench_evaluate(env,self.low_policy,self.eval_episodes)
            if self.video_path:
                height, width, layers  = list(frames.values())[0][0].shape
                size = (width,height)
                fps = 15
                
                for k,v in frames.items():
                    file_name = str(self.n_update) + "_" + k + ".mp4" 
                    video_path = os.path.join(self.video_path,file_name)
                    
                    print("saving: ",file_name)
                    save_video(video_path,v)
            return success_rate
        


        

    def save(self,path):
        # date_time = datetime.now().strftime('%m-%d_%H:%M')
        save_path = os.path.join(self.weights_path,path)
        self.low_policy.save(save_path)

    def record(self,info):
        mse_loss = info['decoder/mse_loss']
        ce_loss = info['decoder/ce_loss']
        if self.wandb_record:
            self.wandb_logger.log({
                "mse loss": mse_loss,
                "ce loss": ce_loss
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


