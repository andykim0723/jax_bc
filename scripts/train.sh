# python train.py --task 'd4rl-halfcheetah' --policy 'bc' 
# python train.py --task 'd4rl-hopper' --policy 'bc' 
CUDA_VISIBLE_DEVICES=1 python train.py --task 'rlbench-pick_and_lift' --policy 'bc' 

