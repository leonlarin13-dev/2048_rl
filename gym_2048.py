import gymnasium as gym
from gymnasium import spaces
import numpy as np
from logic import *  # Your extended logic.py with valid move handling

class Game2048Env(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = spaces.Box(low=0, high=2**16, shape=(4, 4), dtype=np.int32)

        self.render_mode = render_mode
        self.max_steps = max_steps

        self.mat = None
        self.done = False
        self.steps = 0
        self.invalid_move_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mat = start_game()
        self.done = False
        self.steps = 0
        self.invalid_move_count = 0

        if self.render_mode == 'human':
            self.render()

        return np.array(self.mat, dtype=np.int32), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Cannot call step() on a terminated environment. Please call reset().")

        self.steps += 1

        # Perform move
        if action == 0:
            new_mat, changed, merge_reward = move_up(self.mat)
        elif action == 1:
            new_mat, changed, merge_reward = move_down(self.mat)
        elif action == 2:
            new_mat, changed, merge_reward = move_left(self.mat)
        elif action == 3:
            new_mat, changed, merge_reward = move_right(self.mat)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Handle invalid move
        if not changed:
            self.invalid_move_count += 1
            terminated = self.invalid_move_count >= 10  # Limit invalids
            truncated = self.steps >= self.max_steps
            self.done = terminated or truncated
            return (
                np.array(self.mat, dtype=np.int32),
                -1.0,  # penalty for invalid action
                terminated,
                truncated,
                self._info("INVALID")
            )

        # Valid move: reset invalid count
        self.invalid_move_count = 0
        self.mat = new_mat
        add_new_2(self.mat)

        # Check game state
        game_state = get_current_state(self.mat)
        terminated = False
        reward = merge_reward / 4.0


                               #tried to steer agent into keeping highest tile in the corner didnt work
        #highest_tile = np.max(self.mat)
        #corner = (0, 0)
        #row, col = np.argwhere(self.mat == highest_tile)[0]
        #distance_penalty = (abs(row - corner[0]) + abs(col - corner[1])) * 2
        #reward += 40 - distance_penalty

        if game_state == "WON":
            reward += 1000.0
            terminated = True
        elif game_state == "LOST":
            #reward -= 10.0
            terminated = True

        truncated = self.steps >= self.max_steps
        self.done = terminated or truncated

        if self.render_mode == 'human':
            self.render()

        return (
            np.array(self.mat, dtype=np.int32),
            reward,
            terminated,
            truncated,
            self._info(game_state)
        )

    def _info(self, state):
        return {
            "valid_actions": self.get_valid_actions(),
            "score": int(np.sum(self.mat)),  # or track merge reward separately
            "state": state,
            "steps": self.steps,
            #"merge_reward": merge_reward,
        }


    #used to block invalid move loops in training or evaluating
    def get_valid_actions(self):
        valid = []
        for action in range(4):
            board_copy = [row[:] for row in self.mat]
            if action == 0:
                _, changed, _ = move_up(board_copy)
            elif action == 1:
                _, changed, _ = move_down(board_copy)
            elif action == 2:
                _, changed, _ = move_left(board_copy)
            elif action == 3:
                _, changed, _ = move_right(board_copy)
            if changed:
                valid.append(action)
        return valid

    def render(self):
        print("+------" * 4 + "+")
        for row in self.mat:
            print("".join(f"|{str(num).center(6) if num != 0 else ' '.center(6)}" for num in row) + "|")
        print("+------" * 4 + "+\n")

    def close(self):
        pass

    def _state_to_key(self):
        return tuple(tuple(row) for row in self.mat)
