import os
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from torch.distributions import Categorical
from skimage import transform
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

def preprocess(obs, is_new_episode, stacked_frames, stack_size=4, img_shape=(84, 84)):
    processed_obs = np.mean(obs, axis=2)
    processed_obs = transform.resize(processed_obs, img_shape)
    processed_obs = processed_obs / 255.0

    if is_new_episode:
        stacked_frames = deque([np.zeros(img_shape, dtype=np.float32) for _ in range(stack_size)], maxlen=4)
        for _ in range(stack_size):
            stacked_frames.append(processed_obs)
    else:
        stacked_frames.append(processed_obs)

    stacked_state = np.stack(stacked_frames, axis=0)
    return stacked_state, stacked_frames

env = gym.make('ALE/SpaceInvaders-v5', render_mode=None)

class ActorCritic(nn.Module):
    def __init__(self, n_channels, n_actions):
        super(ActorCritic, self).__init__()
        # Convolutional layers and fully connected layers as before
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(n_channels), 512),
            nn.ReLU(),
        )
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

    def _feature_size(self, n_channels):
        return self.conv(torch.zeros(1, n_channels, 84, 84)).view(1, -1).size(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

class PPOAgent:
    def __init__(self, n_channels, n_actions):
        self.gamma = 0.99
        self.clip_param = 0.2
        self.ppo_epochs = 4
        self.mini_batch_size = 64
        self.model = ActorCritic(n_channels, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.scores = []  # To store raw scores for plotting

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        # Compute GAE as before
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        returns = torch.cat(returns).detach()
        return returns

    def ppo_update(self, states, actions, log_probs, returns, advantages):
        # PPO update as before, with action_probs, value computation and loss calculation
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        for _ in range(self.ppo_epochs):
            action_probs, value = self.model(states)
            dist = Categorical(logits=action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            value_loss = (returns - value).pow(2).mean()

            self.optimizer.zero_grad()
            total_loss = policy_loss + 0.5 * value_loss
            total_loss.backward(retain_graph=True if _ < self.ppo_epochs - 1 else False)
            self.optimizer.step()
        self.scheduler.step()
        
    def train(self, env, episodes):
        total_rewards = []
        batch_scores = []
        batch_rewards = []
        for episode in range(episodes):
            observation, _ = env.reset()
            is_new_episode = True
            stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(4)], maxlen=4)
            state, stacked_frames = preprocess(observation, is_new_episode, stacked_frames)
            done = False
            episode_reward = 0
            raw_score = 0

            # Initialize lists to store episode data
            reward_added = False
            reward2_added = False
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            while not done:
                state = np.expand_dims(state, axis=0)
                state_tensor = torch.FloatTensor(state).to('cpu')
                action_probs, value = self.model(state_tensor)
                dist = Categorical(logits=action_probs)
                action = dist.sample()
                obs, reward, done, _, info = env.step(action.item())
                raw_score += reward
                if raw_score >= 200 and reward_added == False:
                    reward_added = True
                    reward += 100  # Additional reward logic
                    
                if raw_score >= 300 and reward2_added == False:
                    reward2_added = True
                    reward += 300  # Additional reward logic
                    
                next_state, stacked_frames = preprocess(obs, False, stacked_frames)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to('cpu')  # Define next_state_tensor here
                log_prob = dist.log_prob(action).unsqueeze(0)
                episode_reward += reward
                # Collect data
                log_probs.append(log_prob)
                values.append(value)
                states.append(state_tensor)
                actions.append(torch.tensor([action], dtype=torch.long))
                rewards.append(torch.tensor([reward], dtype=torch.float))
                masks.append(torch.tensor([1-done], dtype=torch.float))
                state = next_state

            if not done:
                action_probs, next_value = self.model(next_state_tensor)
            else:
                next_value = torch.tensor([[0]], dtype=torch.float)  # Correctly handle the termination state
                action_probs = torch.zeros_like(next_value)

            values.append(next_value)
            returns = self.compute_gae(next_value, rewards, masks, values)
            values = torch.cat(values[:-1]).detach()

            log_probs = torch.cat(log_probs).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantages = returns - values
            self.ppo_update(states, actions, log_probs, returns, advantages)
            total_rewards.append(episode_reward)
            self.scores.append(raw_score)  # Storing scores for plotting
            batch_scores.append(raw_score)
            batch_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                average_score = sum(batch_scores) / len(batch_scores)
                average_reward = sum(batch_rewards) / len(batch_rewards)
                print(f'Episodes {episode-9}-{episode+1}, Average Raw Score: {average_score}, Average Reward: {average_reward}')
                batch_scores = []  # Reset the scores for the next batch
                batch_rewards = []
                
            if (episode + 1) % 1000 == 0:
                segment_avg = np.mean(total_rewards[episode-999:episode+1])
                raw_avg = np.mean(self.scores[episode-999:episode+1])
                print(f"Average raw score for episodes {episode-999} to {episode}: {raw_avg}")
                print(f"Average reward for episodes {episode-999} to {episode}: {segment_avg}")
                
                
        average_reward = sum(total_rewards) / len(total_rewards)
        average = sum(self.scores) / len(self.scores)
        
        print(f'Average raw score for {episode} episodes: {average}, average reward: {average_reward}')
        
            # Plotting
        plt.plot(range(len(self.scores)), self.scores)
        plt.xlabel('Episode')
        plt.ylabel('Raw Score')
        plt.title('Raw Score Over Time')
        plt.show()


    def save_state(self, filename="ppo_agent_state.pth"):
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, filename)
        print(f"Agent state saved to {filename}")

    def load_state(self, filename="ppo_agent_state.pth"):
        if os.path.isfile(filename):
            state = torch.load(filename)
            self.model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            print(f"Loaded agent state from {filename}")
        else:
            print(f"No saved state found at {filename}, starting from scratch.")

if __name__ == "__main__":
    n_channels = 4  # Number of stacked frames
    n_actions = env.action_space.n
    agent = PPOAgent(n_channels=n_channels, n_actions=n_actions)
    agent.load_state()  # Attempt to load a previously saved state
    try:
        agent.train(env, episodes=6000)
    finally:
        agent.save_state()
        env.close()
