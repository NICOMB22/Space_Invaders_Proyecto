import os
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Function to save hyperparameters
def save_hyperparameters(params, filename):
    with open(filename, 'w') as f:
        json.dump(params, f)

# Function to load hyperparameters
def load_hyperparameters(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    else:
        # Default hyperparameters if no file exists
        return {'learning_rate': 2.5e-4}  # Example default learning rate

# Custom learning rate schedule
def linear_schedule(initial_value):
    def func(progress):
        return initial_value * (1 - progress)
    return func

# Callback to log average reward and length
class RewardLoggingCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_len = []
        
    def _on_step(self):
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_len.append(info['episode']['l'])
                    if len(self.episode_rewards) % self.check_freq == 0:
                        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                        avg_len = sum(self.episode_len) / len(self.episode_len)
                        with open("ppo_training_log.txt", "a") as f:
                            f.write(f"Average Reward after {len(self.episode_rewards)} episodes: {avg_reward}\n")
                            f.write(f"Average Len after {len(self.episode_len)} episodes: {avg_len}\n")
                        self.episode_rewards = []  # Reset the rewards list after logging
                        self.episode_len = []
        return True

# Environment setup
env_id = 'ALE/SpaceInvaders-v5'
env = make_atari_env(env_id, n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

# Load or set default hyperparameters
hyperparameters_file = "./models/ppo_space_invaders_hyperparameters.json"
hyperparameters = load_hyperparameters(hyperparameters_file)

# Model setup
model_path = "./models/final_ppo_space_invaders_model.zip"
if os.path.exists(model_path):
    model = PPO.load(model_path, env=env)
else:
    # Initialize a new model if no saved model exists
    model = PPO("CnnPolicy", env, verbose=1, learning_rate=linear_schedule(hyperparameters['learning_rate']))

# Callbacks
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='ppo_spaceinvaders')
reward_logging_callback = RewardLoggingCallback(check_freq=4)

# Training
model.learn(total_timesteps=2000000, callback=[checkpoint_callback, reward_logging_callback])

# Save hyperparameters and model
save_hyperparameters({'learning_rate': hyperparameters['learning_rate']}, hyperparameters_file)
model.save(model_path)

# Evaluation
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Cleanup
env.close()
