import jax.numpy as jnp
import flax.linen as nn
import pickle as pkl
import numpy as np
import os 
from jax_resnet import pretrained_resnet

from PIL import Image
def feature_extraction_test():        
    resnet18, variables = pretrained_resnet(18)
    model = resnet18()
    fe_model = nn.Sequential(model.layers[0:-1])
    out = fe_model.apply(variables,
                    jnp.ones((1, 224, 224, 3)),  # ImageNet sized inputs.
                    mutable=False)

    print(out.shape)

def dataset_test():


    path = "data/pick_and_lift/variation0/episodes/episode0/low_dim_obs.pkl"
    with open(path,'rb') as f:
        episode = pkl.load(f)
        print(episode.__len__())
        episode = episode._observations 

        test = episode[0]
        low_dim_state = test.task_low_dim_state # (6,)
        print(test.task_low_dim_state)
        exit()
        low_dim_data = [] if test.gripper_open is None else [[test.gripper_open]]
        # for data in [test.joint_velocities, test.joint_positions,
        #              test.joint_forces,
        #              test.gripper_pose, test.gripper_joint_positions,
        #              test.gripper_touch_forces, test.task_low_dim_state]:
        print(episode[0].get_low_dim_data())
        exit()
        observations = np.concatenate([obs.get_low_dim_data()[np.newaxis,:] for obs in episode],axis=0)
        print(observations.shape)
        exit()


        exit()
        episode_ts = episode.__len__()
        for dr in range(episode_ts):
            obs = episode.__getitem__(0).get_low_dim_data()
            print(obs.shape)


def rlbench_env_test():
    import gym
    import rlbench.gym

    env = gym.make('pick_and_lift-vision-v0')
    env.reset()

    # Alternatively, for vision:
    # env = gym.make('reach_target-vision-v0')

    path = "data/pick_and_lift/variation0/episodes/episode0/low_dim_obs.pkl" 
    with open(path,'rb') as f:
        observations = pkl.load(f)._observations
 
    steps = observations.__len__()
    for i in range(steps):
        # if i % episode_length == 0:
        #     print('Reset Episode')
        #     obs = env.reset()
        obs = observations[i]
        action = np.append(obs.joint_velocities,[obs.gripper_open])  
        obs, reward, terminate, _ = env.step(action)
        Image.fromarray(obs['front_rgb']).save('test.png')
        # env.render()  # Note: rendering increases step time.

    print('Done')
    env.close()
if __name__ == '__main__':
    # feature_extraction_test()
    dataset_test()
    # rlbench_env_test()