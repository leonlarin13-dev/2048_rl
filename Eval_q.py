import pickle
from collections import defaultdict, Counter
import numpy as np
from gym_2048 import Game2048Env  # Your Gymnasium wrapper

ACTIONS_MAP = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

def state_to_str(state):
    return tuple(state.flatten())

def main(render=True, episodes=10):
    q_table_file = 'q_table_2048_plotting.pkl'       #here file name
    with open(q_table_file, 'rb') as f:
        loaded = pickle.load(f)
    q_table = defaultdict(float, loaded)
    print(f"Loaded Q-table from '{q_table_file}'.")

    env = Game2048Env()
    total_rewards = 0
    max_tiles = []
    max_tile_counts = Counter()

    for ep in range(episodes):
        state, _ = env.reset()
        state_str = state_to_str(state)
        done = False
        episode_reward = 0

        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                print("No valid actions available. Ending episode.")
                break

            # Get Q-values for all actions, ignoring invalid ones
            q_values = [q_table[(state_str, a)] if a in valid_actions else float('-inf') for a in range(4)]
            action = max(valid_actions, key=lambda a: q_values[a])

            if render:
                env.render()
                print(f"Chosen action: {ACTIONS_MAP[action]}")
                print(f"Q-values: {q_values}")
                #print(f"Valid actions: {[ACTIONS_MAP[a] for a in valid_actions]}\n")

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state_str = state_to_str(next_state)
            episode_reward += reward

            if reward == -1.0:
                print("Invalid move detected (no change in state).\n")

        final_max_tile = np.max(env.mat)
        max_tiles.append(final_max_tile)
        max_tile_counts[final_max_tile] += 1
        total_rewards += episode_reward

        print(f"Episode {ep+1}: Total Reward = {episode_reward:.2f}, Max Tile = {final_max_tile}\n")
        if render:
            print("Final Board:")
            env.render()

    avg_reward = total_rewards / episodes
    avg_max_tile = sum(max_tiles) / episodes

    print(f"\nEvaluation over {episodes} episode(s):")
    print(f"Average Reward:   {avg_reward:.2f}")
    print(f"Average Max Tile: {avg_max_tile:.2f}\n")

    print("Max Tile Frequencies:")
    for tile, count in sorted(max_tile_counts.items()):
        print(f"{tile:>5}: {count} time(s)")

    env.close()

if __name__ == "__main__":
    # Set render=False for silent batch evaluation
    main(render=True, episodes=5)
