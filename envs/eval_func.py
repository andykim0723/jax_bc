import RLBench.rlbench.gym
from PIL import Image
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

    episode_length = 300
    for i in range(num_episodes):
        
        obs = env.reset()
        state = obs['state'][-6:] # last 6 dim represent task_low_dim_state 
        for j in range(episode_length):

            action = policy.predict(state)
            obs, reward, terminate, _ = env.step(action)
            state = obs['state'][-6:]
            img = obs['front_rgb']
            Image.fromarray(img).save('test.png')

            if terminate:
                break
        print(f"episode{i}: reward {reward}, terminated: {terminate} ")

    # return rewards
