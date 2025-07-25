import numpy as np
from game2048.gym_2048 import Game2048Env

EPISODES = 100
MAX_STEPS = 5000 

def run_random_agent(episodes=EPISODES):
    env = Game2048Env()
    total_rewards = 0
    max_tiles = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < MAX_STEPS:
            action = np.random.randint(0, 4)  # Random action: 0–3
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1

        total_rewards += episode_reward
        max_tiles.append(np.max(env.mat))

        print(f"🎮 Episode {ep + 1}: Reward = {episode_reward:.2f}, Max Tile = {np.max(env.mat)}")

    avg_reward = total_rewards / episodes
    avg_max_tile = sum(max_tiles) / episodes

    print("\nRandom Policy Evaluation:")
    print(f"Average Reward:   {avg_reward:.2f}")
    print(f"Average Max Tile: {avg_max_tile:.2f}")
    env.close()

if __name__ == "__main__":
    run_random_agent()
