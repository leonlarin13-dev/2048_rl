# 2048_rl
# 2048 RL  
Reinforcement learning agents trained to play the game **2048** using **Q-Learning** and **Monte Carlo Control**.

## Requirements

- Python 3.12  
- Install dependencies via:

```bash
pip install -r requirements.txt
```

## Training Agents

Training scripts are located in the `train` directory.

Each script will output a `.pkl` file containig the Q-values of the trained models.

## Evaluating Agents

Evaluation scripts are located in the `eval` directory.

## Project Structure

```
.
├── eval/
│   ├── eval_montecarlo.py
│   ├── eval_qlearning.py
│   └── eval_montecarlo.py
├── game2048/
│   ├── gym_2048.py
│   ├── logic.py
│   └── terminal_play2048.py
├── train/
│   ├── train_qlearning.py
│   └── train_montecarlo.py
├── requirements.txt
└── README.md
```

---
