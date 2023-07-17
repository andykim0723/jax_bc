# jax_bc

**jax dependencies**

need to match cuda, cudnn version when installing jax,flax:

jax==0.4.6 \
jaxlib==0.4.6+cuda11.cudnn82 \
flax==0.7.0 


---

**headless mode for RLBench**
```
python jaxbc/utils/startx.py
export DISPLAY=:0.0                                            
nohup sudo X & 
```