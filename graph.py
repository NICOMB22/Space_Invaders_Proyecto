import pandas as pd
import matplotlib.pyplot as plt
import re

# Read the log files
dqn_log_file = 'training_log.txt'
ppo_log_file = 'ppo_training_log.txt'
random_log_file = 'random_training_log.txt'

# Initialize lists to store data for DQN and PPO
dqn_episodes = []
dqn_average_rewards = []
ppo_episodes = []
ppo_average_rewards = []

# Regular expressions to extract data for DQN and PPO
reward_pattern = re.compile(r'Average Reward after (\d+) episodes: ([\d.]+)')

# To handle episode number restarts for DQN and PPO
current_offset_dqn = 0
last_episode_dqn = 0
current_offset_ppo = 0
last_episode_ppo = 0

def parse_log_file(log_file, episodes_list, rewards_list, current_offset, last_episode):
    with open(log_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            reward_match = reward_pattern.search(line)
            
            if reward_match:
                episode = int(reward_match.group(1))
                average_reward = float(reward_match.group(2))
                if episode <= last_episode:
                    current_offset += last_episode
                actual_episode = episode + current_offset
                episodes_list.append(actual_episode)
                rewards_list.append(average_reward)
                last_episode = episode
                
    return episodes_list, rewards_list, current_offset, last_episode

# Parse DQN and PPO log files
dqn_episodes, dqn_average_rewards, current_offset_dqn, last_episode_dqn = parse_log_file(dqn_log_file, dqn_episodes, dqn_average_rewards, current_offset_dqn, last_episode_dqn)
ppo_episodes, ppo_average_rewards, current_offset_ppo, last_episode_ppo = parse_log_file(ppo_log_file, ppo_episodes, ppo_average_rewards, current_offset_ppo, last_episode_ppo)

# Initialize lists to store data for Random agent
random_episodes = []
random_rewards = []

# Regular expression to extract data for Random agent
random_pattern = re.compile(r'Episode (\d+): Total Reward = ([\d.]+)')

# Parse Random agent log file
with open(random_log_file, 'r') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        
        random_match = random_pattern.search(line)
        
        if random_match:
            episode = int(random_match.group(1))
            reward = float(random_match.group(2))
            random_episodes.append(episode)
            random_rewards.append(reward)

# Create DataFrames
dqn_data = {
    'Episode': dqn_episodes,
    'Average Reward': dqn_average_rewards
}
ppo_data = {
    'Episode': ppo_episodes,
    'Average Reward': ppo_average_rewards
}
random_data = {
    'Episode': random_episodes,
    'Average Reward': random_rewards
}

df_dqn = pd.DataFrame(dqn_data)
df_ppo = pd.DataFrame(ppo_data)
df_random = pd.DataFrame(random_data)

# Debug output to check the data
print(df_dqn)
print(df_ppo)
print(df_random)

# Function to plot graphs for a given range of episodes
def plot_graphs(df, title_suffix):
    # Plot Average Reward over Episodes
    plt.figure(figsize=(10, 6))
    plt.plot(df['Episode'], df['Average Reward'], label='Average Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title(f'Average Reward over Episodes {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Average Length over Episodes
    if 'Average Length' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Episode'], df['Average Length'], label='Average Length', color='orange')
        plt.xlabel('Episodes')
        plt.ylabel('Average Length')
        plt.title(f'Average Length over Episodes {title_suffix}')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot Average Reward over Average Length
    if 'Average Length' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Average Length'], df['Average Reward'], label='Reward vs Length', color='red')
        plt.xlabel('Average Length')
        plt.ylabel('Average Reward')
        plt.title(f'Average Reward vs Average Length {title_suffix}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Plot graphs for every 20,000 episodes for DQN
chunk_size = 20000
max_episode_dqn = df_dqn['Episode'].max()
# for start in range(0, max_episode_dqn, chunk_size):
#     end = start + chunk_size
#     df_chunk = df_dqn[(df_dqn['Episode'] > start) & (df_dqn['Episode'] <= end)]
#     if not df_chunk.empty:
#         plot_graphs(df_chunk, f'({start + 1} - {end})')

# Plot graphs for every 20,000 episodes for PPO
max_episode_ppo = df_ppo['Episode'].max()
# for start in range(0, max_episode_ppo, chunk_size):
#     end = start + chunk_size
#     df_chunk = df_ppo[(df_ppo['Episode'] > start) & (df_ppo['Episode'] <= end)]
#     if not df_chunk.empty:
#         plot_graphs(df_chunk, f'({start + 1} - {end})')

# Plot overall graphs for DQN and PPO
# plot_graphs(df_dqn, '(Overall)')
# plot_graphs(df_ppo, '(Overall)')

# Plot overall graph for Random agent with a descriptive title
plt.figure(figsize=(10, 6))
plt.plot(df_random['Episode'], df_random['Average Reward'], label='Random Agent', color='red')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward over Episodes (Random Agent)')
plt.legend()
plt.grid(True)
plt.show()

# Plot comparative graph for DQN, PPO, and Random agent
plt.figure(figsize=(10, 6))
plt.plot(df_dqn['Episode'], df_dqn['Average Reward'], label='DQN', color='blue')
plt.plot(df_ppo['Episode'], df_ppo['Average Reward'], label='PPO', color='green')
plt.plot(df_random['Episode'], df_random['Average Reward'], label='Random', color='red')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward Comparison')
plt.legend()
plt.grid(True)
plt.show()
