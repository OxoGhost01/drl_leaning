import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from huggingface_sb3 import package_to_hub

model = PPO.load("UNIT-1/ppo_lunarlander-v2.2")
model_name = "ppo_lunarlander-v2"

## TODO: Define a repo_id
## repo_id is the id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
repo_id = "OxoGhost/ppo-LunarLander-v2"

# TODO: Define the name of the environment
env_id = "LunarLander-v2"

# Create the evaluation env and set the render_mode="rgb_array"
eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])


# TODO: Define the model architecture we used
model_architecture = "PPO"

## TODO: Define the commit message
commit_message = "Lesgo!"

# method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
package_to_hub(model=model, # Our trained model
                model_name=model_name, # The name of our trained model 
                model_architecture=model_architecture, # The model architecture we used: in our case PPO
                env_id=env_id, # Name of the environment
                eval_env=eval_env, # Evaluation Environment
                repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
                commit_message=commit_message)