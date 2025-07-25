import os
import pickle
import random
import numpy as np
from collections import defaultdict
from game2048.gym_2048 import Game2048Env  

# Hyperparameters
epsilon = 1.0
epsilon_min = 0.2
epsilon_decay = 0.99995
episodes = 50000
max_steps_per_episode = 500
gamma = 0.98

q_table_file = 'mc_q_table_2048.pkl'

def state_to_str(state):
    # Use a consistent tuple-based key
    return tuple(state.flatten())

def choose_action(state_str, q_table, epsilon, valid_actions):
    if not valid_actions:
        return None  # No valid move
    if random.random() < epsilon:
        return random.choice(valid_actions)
    else:
        q_vals = [q_table[(state_str, a)] for a in valid_actions]
        max_q = max(q_vals)
        best_actions = [a for a in valid_actions if q_table[(state_str, a)] == max_q]
        return random.choice(best_actions)

def generate_episode(env, q_table, epsilon):
    episode = []
    state, _ = env.reset()
    state_str = state_to_str(state)

    for _ in range(max_steps_per_episode):
        valid_actions = env.get_valid_actions()
        action = choose_action(state_str, q_table, epsilon, valid_actions)

        if action is None:
            break  # No valid moves, end episode

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state_str = state_to_str(next_state)
        done = terminated or truncated

        # Always store experience, even with negative reward
        episode.append((state_str, action, reward))

        state_str = next_state_str

        if done:
            break
    return episode

def main():
    global epsilon
    env = Game2048Env()

    if os.path.exists(q_table_file):
        with open(q_table_file, 'rb') as f:
            loaded = pickle.load(f)
        q_table = defaultdict(float, loaded)
        returns = defaultdict(list)
        print("Loaded existing Q-table.")
    else:
        q_table = defaultdict(float)
        returns = defaultdict(list)
        print("Starting fresh.")

    for ep in range(episodes):
        episode = generate_episode(env, q_table, epsilon)

        G = 0
        visited = set()
        for state_str, action, reward in reversed(episode):
            G = reward + gamma * G
            if (state_str, action) not in visited:
                returns[(state_str, action)].append(G)
                q_table[(state_str, action)] = np.mean(returns[(state_str, action)])
                visited.add((state_str, action))

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (ep + 1) % 100 == 0:
            total_reward = sum(x[2] for x in episode)
            #reward_log.append((ep + 1, total_reward))
            print(f"Episode {ep+1:>5}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.4f}")


    with open(q_table_file, 'wb') as f:
        pickle.dump(dict(q_table), f)
    print(f"\nQ-table saved to `{q_table_file}`.")

    #with open('reward_log_2048.pkl', 'wb') as f:
    #    pickle.dump(reward_log, f)
    #print("Reward log saved to `reward_log_2048.pkl`.")


    env.close()

if __name__ == "__main__":
    #reward_log = []
    main()
