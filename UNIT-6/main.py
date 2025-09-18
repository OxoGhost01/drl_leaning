import os

import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env


env_id = "PandaReachDense-v3"

# Create the env
env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape
a_size = env.action_space

def get_info(env):
    print("Observation Space : ", env.observation_space)
    print("Sample observation : ", env.observation_space.sample())  # Get a random observation

    print("\nAction Space : ", env.action_space)
    print("Action Sample : ", env.action_space.sample())  # Take a random action

env = make_vec_env(env_id, n_envs=4)

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = A2C("MultiInputPolicy", env, verbose=1)
model.learn(1_000_000)

model.save("a2c-PandaReachDense-v3")
env.save("vec_normalize.pkl")


