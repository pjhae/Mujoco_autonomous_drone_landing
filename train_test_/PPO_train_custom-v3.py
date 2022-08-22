import gym
import torch as th
import torch.nn as nn
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack


def lin_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func


date = "0815"
trial = "A"

checkpoint_on_event = CheckpointCallback(
    save_freq=1,
    save_path='./save_model_'+date+'/'+trial,
    verbose=2,
    name_prefix='Hexy_model_'+date+trial
)

event_callback = EveryNTimesteps(
    n_steps=int(1e4),  # every n_steps, save the model
    callback=checkpoint_on_event
)


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                n_input_channels = subspace.shape[0]
                extractors[key] = nn.Sequential(nn.Conv2d(n_input_channels, 16, 8, stride=4, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 4, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                )

                total_concat_size += 4*4*4
                
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 6)
                total_concat_size += 6

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


env = make_vec_env("Drone-v2", n_envs=1)
# env = VecFrameStack(env, n_stack=3,  channels_order = "first")

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
)

model = PPO("MultiInputPolicy", env=env , device = 'cuda', policy_kwargs=policy_kwargs, verbose=1,  tensorboard_log='./hexy_tb_log_'+ date,
            learning_rate=lin_schedule(3e-4, 3e-6), clip_range=lin_schedule(0.3, 0.1),
            n_epochs=10, ent_coef=1e-4, batch_size=256*4, n_steps=256)

model.learn(total_timesteps=100000000,
		callback=event_callback,  # every n_steps, save the model.
		tb_log_name='hexy_tb_'+date+trial
		# ,reset_num_timesteps=False   # if you need to continue learning by loading existing model, use this option.
		)

model.save("Hexy_model")
del model # remove to demonstrate saving and loading
model = PPO.load("Hexy_model")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    
    env.render()
