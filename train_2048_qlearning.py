import numpy as np
import random
import os
import pickle
from collections import defaultdict
import gymnasium as gym
from gym_2048 import Game2048Env

# Q-learning hyperparameters
alpha = 0.1
gamma = 0.98
epsilon = 1.0
epsilon_min = 0.2
epsilon_decay = 0.99995
episodes = 50000
max_steps_per_episode = 500

q_table_file = 'q_table_2048_plotting.pkl'

def state_to_str(state):
    # Use tuple for faster hashing and uniqueness
    return tuple(state.flatten())

def choose_action(state_str, q_table, valid_actions, epsilon):
    if not valid_actions:
        return random.randint(0, 3)  # fallback if no valid actions

    if random.uniform(0, 1) < epsilon:
        return random.choice(valid_actions)

    # Choose the best Q-value among valid actions
    q_values = [q_table[(state_str, a)] if a in valid_actions else float('-inf') for a in range(4)]
    max_q = max(q_values)
    best_actions = [a for a, q in enumerate(q_values) if q == max_q]
    return random.choice(best_actions)

def main():
    global epsilon
    env = Game2048Env()

    # Load Q-table if it exists
    if os.path.exists(q_table_file):
        with open(q_table_file, 'rb') as f:
            loaded = pickle.load(f)
        q_table = defaultdict(float, loaded)
        print("Loaded existing Q-table.")
    else:
        q_table = defaultdict(float)
        print("Starting fresh Q-table.")

    for ep in range(episodes):
        state, _ = env.reset()
        state_str = state_to_str(state)
        total_reward = 0
        invalid_moves = 0
        episode_score = 0

        for step in range(max_steps_per_episode):
            valid_actions = env.get_valid_actions()
            action = choose_action(state_str, q_table, valid_actions, epsilon)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_str = state_to_str(next_state)

            if reward == -1.0:
                invalid_moves += 1

            episode_score = info.get("score", 0)

            # Q-learning update
            old_value = q_table[(state_str, action)]
            next_max = max(q_table[(next_state_str, a)] for a in range(4))
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[(state_str, action)] = new_value

            state_str = next_state_str
            total_reward += reward

            if terminated or truncated:
                break

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Logging
        if (ep + 1) % 100 == 0:
            reward_log.append((ep + 1, total_reward))
            print(f"Ep {ep+1:5d} | Total Reward: {total_reward:6.2f} | Final Score: {episode_score} | "
                  f"Invalid Moves: {invalid_moves} | Epsilon: {epsilon:.4f}")

        # Save checkpoint every 5000 episodes
        if (ep + 1) % 5000000 == 0:
            checkpoint_file = f'q_table_checkpoint_forplot_{ep+1}.pkl'
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(dict(q_table), f)
            print(f"Saved checkpoint to {checkpoint_file}")

    # Final Q-table save
    with open(q_table_file, 'wb') as f:
        pickle.dump(dict(q_table), f)
    print(f"Training complete. Q-table saved to {q_table_file}")

    with open('reward_log_qlearning.pkl', 'wb') as f:
        pickle.dump(reward_log, f)
    print("Reward log saved to `reward_log_qlearning.pkl`.")

    env.close()

if __name__ == "__main__":
    reward_log = []
    main()
