import os
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
        
        self.log_interval = cfg['train']['log_interval']
        self.save_interval = cfg['train']['save_interval']
        self.eval_interval = cfg['eval']['eval_interval']

        # time
        self.prepare_run()


    def run(self,replay_buffer,env):  
        #
        for step in range(int(self.cfg['train']['steps'])):
            replay_data = replay_buffer.sample(batch_size = self.batch_size)
            info = self.low_policy.update(replay_data)
            self.n_update += 1

            if (self.n_update % self.log_interval) == 0:
                print(self.n_update,info['decoder/mse_loss'])
                # self.record(self.n_update,info)
        
            if (self.n_update % self.save_interval) == 0:
                self.save()
            
            if (self.n_update % self.eval_interval) == 0:
                print(f'n_step: {self.n_update}. start evaluation..')
                rewards = self.evaluate(env)
                print("rewards: ", np.mean(rewards))
 
    
    def record(self,step,info):
        raise NotImplementedError
        loss = info['decoder/mse_loss']
        save_path = self.cfg['save_path']
        print(step,loss)

    def save(self):
        save_path = os.path.join('logs','test')
        self.low_policy.save(save_path)
        # raise NotImplementedError
        # for key, save_path in self.cfg["save_paths"].items():
        #     getattr(self, key).save(save_path)   


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

