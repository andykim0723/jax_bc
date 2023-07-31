
# Setup


**jax dependencies**

need to match cuda, cudnn version when installing jax,flax. I used:

```
pip install jax==0.4.6 
pip install jaxlib==0.4.6+cuda11.cudnn82 
pip install flax==0.7.0 
```


**other packages**

```
# jax-resnet
pip install --upgrade git+https://github.com/n2cholas/jax-resnet.git

# d4rl
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

pip install wandb

```

# Usage

**headless mode for RLBench**
```
python jaxbc/utils/startx.py
export DISPLAY=:0.0                                            
nohup sudo X & 
```
**run command**
```
python bc.py

# verify BC model on d4rl
python d4rl.py

```

## validation

- [X] D4RL gym mujoco 

* halfcheetah: BC(1e5 steps): 12561.03 expert 12135.0 

* hopper: BC(5e4 steps): 3645.24 expert: 3234.3

- [ ] D4RL carla
* carla-lane-v0:
* carla-town-v0:

- [ ] RLBench: 

## todo
1. ~~log function~~
   1. ~~wandb~~
2. ~~save function~~
   1. ~~save jax model~~
   2. ~~load jax model~~
3. additional env
   1. ~~d4rl hopper~~ 
   2. carla
   3. RLBench
4. online evaluation
5. config 정리
