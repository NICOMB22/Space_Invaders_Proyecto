import os
import json
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import cv2
import time
from gymnasium.wrappers import TimeLimit

class RewardLoggingCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.total_reward = 0
        self.episode_rewards = []
        self.episode_len = []
        self.episode_count = 0
        
    def _on_step(self):
        info = self.locals['infos']
        reward = self.locals['rewards'][0]  # Assuming single environment
        #if self.locals['dones'][0] == True:
            #print(self.locals['dones'][0])
            #print(self.locals.keys())
            #print(self.locals['infos'])
            #print('done')
        self.total_reward += reward
        #print(info[0])
        if info[0]['lives'] == 0:
            ep_reward = info[0]['episode']['r']
            ep_len = info[0]['episode']['l']
            self.episode_rewards.append(ep_reward)
            self.episode_len.append(ep_len)
            self.episode_count += 1

            # Calculate and log the average reward only after certain number of episodes
            if self.episode_count % self.check_freq == 0:
                model = self.locals['self']
                exploration_rate = model.exploration_rate
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                avg_len = sum(self.episode_len) / len(self.episode_len)
                with open("training_log.txt", "a") as f:
                    f.write(f"Average Reward after {self.episode_count} episodes: {avg_reward}\n")
                    f.write(f"Average Len after {self.episode_count} episodes: {avg_len}\n")
                    f.write(f"Current Exploration Rate after {self.episode_count} episodes: {exploration_rate}\n")
                self.episode_rewards = []  # Reset the rewards list after logging
                self.episode_len = []

        return True

# Helper functions for epsilon
def save_epsilon(epsilon, filename):
    with open(filename, 'w') as f:
        json.dump({'epsilon': epsilon}, f)

def load_epsilon(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data['epsilon']
    return 1.0  # Default to 1 if no file exists

env_id = 'ALE/SpaceInvaders-v5'
env = make_atari_env(env_id, n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

model_path = "./models/final_dqn_space_invaders_model.zip"
epsilon_file = "./models/dqn_space_invaders_epsilon.json"

initial_epsilon = 1.0  # Start from 1.0
final_epsilon = 0.01   # Decrease to 0.01
exploration_fraction = 0.2  

# Load epsilon value or set to default if not existing
epsilon_value = load_epsilon(epsilon_file) if os.path.exists(epsilon_file) else 1.0

if os.path.exists(model_path):
    model = DQN.load(model_path, env=env, exploration_initial_eps = 0.01, exploration_final_eps=final_epsilon,
                exploration_fraction=exploration_fraction)
else:
    print('nuevo nosirvenada')
    # New model with exploration parameters set to decrease
    model = DQN("CnnPolicy", env, verbose=1,
                exploration_initial_eps=initial_epsilon,
                exploration_final_eps=final_epsilon,
                exploration_fraction=exploration_fraction)

# Callbacks for model saving
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='dqn_spaceinvaders')
reward_logging_callback = RewardLoggingCallback(check_freq=4)

# Training the model
model.learn(total_timesteps=1000000, callback=[checkpoint_callback, reward_logging_callback])

# Save the model and the epsilon value
model.save(model_path)
save_epsilon(model.exploration_rate, epsilon_file)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Cleanup
env.close()

'''

# Create and preprocess the environment
def make_env(env_id):
    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    return env

def find_next_episode_number(video_folder):
    """Find the next available episode number in the video folder."""
    max_episode = -1
    for file_name in os.listdir(video_folder):
        if file_name.startswith("episode") and file_name.endswith(".avi"):
            # Extract the episode number from filenames like 'episode_10.avi'
            episode_num = int(file_name.split('_')[1].split('.')[0])
            if episode_num > max_episode:
                max_episode = episode_num
    return max_episode + 1

def record_episodes(env_id, model_path, video_folder, num_episodes=10):
    env = make_env(env_id)
    model = DQN.load(model_path, env=env)

    # Ensure the video folder exists
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
        
    start_episode = find_next_episode_number(video_folder)
    end_episode = start_episode + num_episodes - 1
    print(f"Episodes will be saved from {start_episode} to {end_episode}")

    for episode in range(start_episode, start_episode + num_episodes):
        obs = env.reset()
        done = False

        video_path = os.path.join(video_folder, f"episode_{episode}.avi")
        first_frame = env.render(mode='rgb_array')
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            frame = env.render(mode='rgb_array')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)  # Save the frame

            time.sleep(0.1)  # Add delay to slow down the gameplay

            current_lives = info[0].get('lives', 0)

            if current_lives != 0:
                done = False
            elif current_lives == 0:
                done = True  

        out.release()  

    env.close()

env_id = 'ALE/SpaceInvaders-v5'
model_path = "./models/final_dqn_space_invaders_model.zip"
video_folder = './videos/'

record_episodes(env_id, model_path, video_folder, num_episodes=10)
'''