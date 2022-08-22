import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack
import sys

date = "0818"
trial = "A"
steps = "20000"

## Make gym environment #

env = make_vec_env("Drone-v1", n_envs=1)

## Path ##

save_path='./save_model_'+date+'/'+trial+'/'

model = PPO.load(save_path+"Hexy_model_"+date+trial+"_"+steps+"_steps")

obs = env.reset()

f = open("actions", 'w')

## Rendering ##

while True:
    action, _states = model.predict(obs)
    print(action, file=f)
    print(action)
    obs, rewards, done, info = env.step(action)
    env.render()

f.close()

