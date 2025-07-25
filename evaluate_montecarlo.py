import pickle
import numpy as np
from collections import defaultdict
from gym_2048 import Game2048Env
import random

MAX_STEPS_PER_EPISODE = 5000

def state_to_str(state):
    """Convert the board state into a hashable tuple format."""
    return tuple(state.flatten())

def choose_greedy_action(state_str, q_table, valid_actions):
    """Choose the best action based on Q-values among valid ones."""
    if not valid_actions:
        return None, [float('-inf')] * 4

    q_vals = [q_table[(state_str, a)] if a in valid_actions else float('-inf') for a in range(4)]
    max_q = max(q_vals)
    best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
    return random.choice(best_actions), q_vals

def render_board(mat):
    """Nicely print the game board."""
    print("\n+------+------+------+------+")
    for row in mat:
        print("|", end="")
        for cell in row:
            print(f" {str(cell).center(4) if cell != 0 else '    '} |", end="")
        print("\n+------+------+------+------+")

def evaluate(q_table_file='mc_q_table_2048.pkl', episodes=100, render=True):
    """Run evaluation episodes using the trained Monte Carlo Q-table."""
    with open(q_table_file, 'rb') as f:
        loaded = pickle.load(f)
    q_table = defaultdict(float, loaded)
    print(f"Loaded Q-table from '{q_table_file}'")

    env = Game2048Env()
    total_rewards = 0
    max_tiles = []

    for ep in range(episodes):
        state, _ = env.reset()
        state_str = state_to_str(state)
        done = False
        episode_reward = 0
        step_count = 0

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            valid_actions = env.get_valid_actions()
            action, q_vals = choose_greedy_action(state_str, q_table, valid_actions)

            if action is None:
                print("No valid actions left. Ending episode.")
                break

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_str = state_to_str(next_state)

            if render:
                print(f"\nStep {step_count + 1}: Action → {['UP', 'DOWN', 'LEFT', 'RIGHT'][action]}")
                print(f"Q-values: {['{:.2f}'.format(q) if q > float('-inf') else ' -inf' for q in q_vals]}")
                render_board(env.mat)

            episode_reward += reward
            state_str = next_state_str
            step_count += 1

        total_rewards += episode_reward
        max_tile = np.max(env.mat)
        max_tiles.append(max_tile)
        print(f"\nEpisode {ep + 1}: Total Reward = {episode_reward:.2f}, Max Tile = {max_tile}")

    avg_reward = total_rewards / episodes
    avg_max_tile = sum(max_tiles) / episodes

    print(f"\nAverage Reward over {episodes} episodes: {avg_reward:.2f}")
    print(f"Average Max Tile over {episodes} episodes: {avg_max_tile:.2f}")
    env.close()

if __name__ == "__main__":
    evaluate(episodes=5, render=True)  # Set to False to suppress board printing
