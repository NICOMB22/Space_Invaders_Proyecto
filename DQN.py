import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from collections import deque

env = gym.make('ALE/SpaceInvaders-v5', render_mode=None)  # Cambiado a None para desactivar la renderización y mejorar el rendimiento

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.fc(x)

def preprocess(obs):
    obs_np = np.array(obs)
    obs_flattened = obs_np.flatten()
    return obs_flattened

class Agent:
    def __init__(self, input_shape, n_actions, gamma=0.99, epsilon=1.0, lr=1e-4, batch_size=32, memory_size=10000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        self.model = DQN(input_shape, n_actions)
        self.target_model = DQN(input_shape, n_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def update_memory(self, transition):
        self.memory.append(transition)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(n_actions)
        state = preprocess(state).astype(np.float32)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        actions = self.model(state)
        return torch.argmax(actions).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Preparar los arrays de NumPy para cada componente del minibatch
        states = np.array([transition[0] for transition in minibatch], dtype=np.float32)
        actions = np.array([transition[1] for transition in minibatch], dtype=np.int64)
        rewards = np.array([transition[2] for transition in minibatch], dtype=np.float32)
        next_states = np.array([transition[3] for transition in minibatch], dtype=np.float32)
        dones = np.array([transition[4] for transition in minibatch], dtype=np.uint8)
        
        # Convertir los arrays de NumPy en tensores de PyTorch
        states_tensor = torch.tensor(states).float()
        next_states_tensor = torch.tensor(next_states).float()
        actions_tensor = torch.tensor(actions).long().unsqueeze(-1)  # Asegúrate de que tenga la forma correcta para gather
        rewards_tensor = torch.tensor(rewards).float()
        dones_tensor = torch.tensor(dones).bool()

        current_q_values = self.model(states_tensor).gather(1, actions_tensor).squeeze(-1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + self.gamma * next_q_values * (~dones_tensor)

        self.optimizer.zero_grad()
        loss = self.criterion(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

input_shape = (210*160*3,)
n_actions = env.action_space.n

agent = Agent(input_shape=input_shape[0], n_actions=n_actions)

agent.model.load_state_dict(torch.load('final_model_weights.pth'))

episodes = 50
for episode in range(episodes):
    observation, info = env.reset()
    state = preprocess(observation)
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        obs, reward, done, _, info = env.step(action)
        next_state = preprocess(obs)
        agent.update_memory((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        agent.replay()

    if episode % 10 == 0:  # Actualizar el modelo objetivo cada 10 episodios
        agent.update_target_model()
    print(f"Episode: {episode+1}, Total reward: {total_reward}, Epsilon: {agent.epsilon}")
    
torch.save(agent.model.state_dict(), 'final_model_weights.pth')
env.close()
