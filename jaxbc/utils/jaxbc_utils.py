import jax
import numpy as np

def random_episodes(num_episodes):
    rng = jax.random.PRNGKey(42)

    episodes = []
    num_episodes = num_episodes
    for i_epi in range(num_episodes):
        num_ts =  np.random.randint(low=5,high=100)
        obs = []
        actions = []
        for ts in range(num_ts):
            img_rng, action_rng = jax.random.split(rng, num=2) # not working, sampling same number each loop
            
            img = np.random.rand(224,224,3)
            action = np.random.randint(low=0,high=5,size=2)

            obs.append(np.expand_dims(img,axis=0))
            actions.append(np.expand_dims(action,axis=0))

        nxt_obs = obs[1:]
        dummy_img = np.zeros(shape=(1,224,224,3))
        nxt_obs.append(dummy_img)
        
        episode = {
            'obs': obs,
            'actions': actions,
            'next_obs': nxt_obs
        }
        episodes.append(episode)
    
    return episodes

