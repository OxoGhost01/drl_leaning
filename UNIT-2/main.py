import numpy as np
import gymnasium as gym
import random
import imageio
import os
import tqdm

import pickle
from tqdm.notebook import tqdm

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="rgb_array")
print("Observation Space : ", env.observation_space)
print("Sample observation : ", env.observation_space.sample())  # Get a random observation