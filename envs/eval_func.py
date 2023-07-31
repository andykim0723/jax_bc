

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
