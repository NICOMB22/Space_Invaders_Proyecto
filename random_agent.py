import gymnasium as gym

def random_agent(episodes=100, log_file="random_training_log.txt"):
    env = gym.make('ALE/SpaceInvaders-v5', render_mode=None)
    total_rewards = []

    with open(log_file, 'w') as f:
        for episode in range(episodes):
            observation, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = env.action_space.sample()  # Select a random action
                obs, reward, done, _, info = env.step(action)
                total_reward += reward

            total_rewards.append(total_reward)
            f.write(f"Episode {episode + 1}: Total Reward = {total_reward}\n")
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()
    return total_rewards

if __name__ == "__main__":
    total_rewards = random_agent(episodes=10000)
    print(f"Average Reward: {sum(total_rewards) / len(total_rewards)}")
