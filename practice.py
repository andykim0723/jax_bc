import jax.numpy as jnp
import flax.linen as nn
import pickle as pkl
import numpy as np
import os 
from jax_resnet import pretrained_resnet
from jaxbc.utils.common import save_video
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

    
    data_path = "/home/andykim0723/jax_bc/data/pick_and_lift_simple/variation0/episodes/episode48/front_rgb"
    pic_paths = []
    for filename in os.listdir(data_path):
        pic_paths.append(filename)

    file_num = len(pic_paths)
    
    pics = [np.array(Image.open(data_path+'/'+f"{i}.png")) for i in range(file_num)]
    save_video('test.mp4',pics)

    exit()
    with open(data_path+'/low_dim_obs.pkl','rb') as f:
        data = pkl.load(f)._observations
        # observations = np.concatenate([obs.task_low_dim_state[np.newaxis,:] for obs in data],axis=0)
        observations = np.concatenate([obs.get_low_dim_data()[np.newaxis,:] for obs in data],axis=0)
        print(observations.shape)
    exit()
    episode = {}
    with open(path,'rb') as f:
        episode = pkl.load(f)
        print(episode.__len__())
        episode = episode._observations 
        joint_positions1 = episode[0].joint_positions
        joint_positions40 = episode[40].joint_positions

        print(joint_positions1)
        print(joint_positions40)
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

    env = gym.make('pick_and_lift_simple-vision-v0')
    env.reset()

    # Alternatively, for vision:
    # env = gym.make('reach_target-vision-v0')

    path = "data/pick_and_lift_simple/variation0/episodes/episode11/low_dim_obs.pkl" 
    with open(path,'rb') as f:  
        observations = pkl.load(f)._observations
    
    steps = observations.__len__()
    prev_joint = np.zeros((7,))
    # prev_joint = env.env.env._scene.robot.arm.get_joint_positions()

    print(len(observations))
    exit()

    for i in range(steps):
        print("hj")
        # if i % episode_length == 0:
        #     print('Reset Episode')
        #     obs = env.reset()
        history_obs = observations[i]
   
        # for absolute_mode=False
        robot_joint_pos = env.env.env._scene.robot.arm.get_joint_positions()
        print(robot_joint_pos)
        print(history_obs.joint_positions)
        exit()


        joint_pos =  history_obs.joint_positions+prev_joint
        action = np.append(joint_pos,[observations[i].gripper_open])  
        obs, reward, terminate, _ = env.step(action)
        Image.fromarray(obs['front_rgb']).save('test.png')

        prev_joint = history_obs.joint_positions

        # env.render()  # Note: rendering increases step time.

    print('Done')
    env.close()

def video_test():
    episode_length = 3
    frames = {f"episode{k}":[] for k in range(episode_length)}
    print(frames)
    exit()
    import cv2
    import glob
    img_array = []
    path = "/home/andykim0723/jax_bc/data/pick_and_lift_simple/variation0/episodes/episode0/front_rgb/"
    for filename in glob.glob(path+'*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
 
    out = cv2.VideoWriter('test.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
def normalization_test():

    path = "data/pick_and_lift_simple/variation0" 

    max_obs = 0
    min_obs = 0
    max_act = 0
    max_act = 0

    episodes = []
    for filename in os.listdir(path+"/episodes"):
        episode_path = os.path.join(path+"/episodes", filename)
        episode = {}
        with open(episode_path+'/low_dim_obs.pkl','rb') as f:
            data = pkl.load(f)._observations
        # observations = np.concatenate([obs.task_low_dim_state[np.newaxis,:] for obs in data],axis=0)
        obs = data[0].get_low_dim_data()

        max_value = 31.173511505126953
        min_value = -21.414112091064453
        data_to_minmax = obs[15:22]
        output = (data_to_minmax - min_value) / (max_value - min_value)
        test = np.concatenate([obs[:15],output,obs[22:]])
    
        print(obs[15:22])
        exit()
        joint_vel=obs.joint_velocities
        joint_pos=obs.joint_positions
        joint_for=obs.joint_forces
        gripr_pos=obs.gripper_pose
        gripr_joi=obs.gripper_joint_positions
        gripr_tou=obs.gripper_touch_forces
        task_low_=obs.task_low_dim_state

        observations = np.concatenate([obs.get_low_dim_data()[np.newaxis,:] for obs in data],axis=0)
        # actions = np.concatenate([np.append(obs.joint_velocities,[obs.gripper_open])[np.newaxis,:] for obs in data],axis=0)
        actions = np.concatenate([np.append(obs.joint_positions,[obs.gripper_open])[np.newaxis,:] for obs in data],axis=0)
        episode['observations'] = observations[:]
        episode['actions'] = actions

        episodes.append(episode)   

    observations = np.concatenate([ep['observations'] for ep in episodes])
    actions = np.concatenate([ep['actions'] for ep in episodes])

    max = 31.173511505126953
    min = -21.414112091064453
    print(np.max(observations),np.min(observations))
    print(np.max(actions),np.min(actions))
    exit()
if __name__ == '__main__':
    # feature_extraction_test()
    dataset_test()
    # rlbench_env_test()
    # video_test()
    # normalization_test()