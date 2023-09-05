import RLBench.rlbench.gym
from PIL import Image
import cv2
import numpy as np
def d4rl_evaluate(env,policy,num_episodes):
    rewards = []
    for n in range(num_episodes):
        obs = env.reset()
        returns = 0

        for t in range(env._max_episode_steps):
            action = policy.predict(obs)
            obs,rew,done,info = env.step(action)
            returns += rew
            if done:
                break

        rewards.append(returns)

    return rewards

def rlbench_evaluate(env,policy,num_episodes):

    episode_length = 220
    num_success = 0
    frames = {f"episode{k}":[] for k in range(num_episodes)}
    for i in range(num_episodes):
        
        obs = env.reset()
        # state = obs['state'][-6:] # last 6 dim represent task_low_dim_state 
        state = obs['state']
    
        # max_value = 31.173511505126953
        # min_value = -21.414112091064453   
        max_value = 82.78092956542969
        min_value = -77.0235595703125
        data_to_minmax = state[15:22]
        output = (data_to_minmax - min_value) / (max_value - min_value)
        observations = np.concatenate([state[:15],output,state[22:]],axis=0)

        for j in range(episode_length):

            action = policy.predict(observations)
            obs, reward, terminate, _ = env.step(action)
            # state = obs['state'][-6:]
            state = obs['state']
            # max_value = 31.173511505126953
            # min_value = -21.414112091064453 
            max_value = 82.78092956542969
            min_value = -77.0235595703125  
            data_to_minmax = state[15:22]
            output = (data_to_minmax - min_value) / (max_value - min_value)
            observations = np.concatenate([state[:15],output,state[22:]],axis=0)
            img = obs['front_rgb']       

            frames[f'episode{i}'].append(img)
            
            if terminate:
                num_success += 1
                break
        print(f"episode{i}: success: {terminate} ")

    succecss_rate = num_success/num_episodes

    return succecss_rate,frames

    return rewards
