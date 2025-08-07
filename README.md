# Easy21 Reinforcement Learning Project

This repository implements a solution to David Silver's Easy21 assignment from his 2015 Reinforcement Learning course. The project demonstrates various reinforcement learning algorithms by creating an environment (Easy21 game) and different RL agents that can interact and play the game.

## 🎮 Easy21 Game

Easy21 is a simplified version of Blackjack with the following rules:

- **Objective**: Get as close to 21 as possible without going over
- **Cards**: Values 1-10, with red cards being negative and black cards positive
- **Actions**: Hit (draw a card) or Stick (end turn)
- **State**: [dealer's showing card, player's current sum]
- **Rewards**: +1 for winning, -1 for losing, 0 for tie
- **Terminal conditions**: Player busts (sum < 1 or > 21) or sticks

## 🤖 Implemented RL Agents

### 1. Monte Carlo Agent (`MCAgent`)
- **Algorithm**: Monte Carlo Control with exploring starts
- **Features**: 
  - Learns from complete episodes
  - Uses first-visit MC for state-action value estimation
  - Implements ε-greedy exploration strategy
- **File**: `ai/monte_carlo.py`

### 2. Sarsa(λ) Agent (`SarsaAgent`)
- **Algorithm**: Backward-view Sarsa(λ) with eligibility traces
- **Features**:
  - Temporal difference learning with eligibility traces
  - Configurable λ parameter (0-1)
  - Online learning with immediate updates
- **File**: `ai/sarsa_lambda.py`

### 3. Sarsa(λ) with Linear Function Approximation (`SarsaApproxAgent`)
- **Algorithm**: Sarsa(λ) with linear function approximation
- **Features**:
  - Uses feature vectors to represent state-action pairs
  - 36-dimensional feature space
  - Gradient descent weight updates
  - Handles continuous state spaces efficiently
- **File**: `ai/sarsa_lambda.py`

### 4. Random Agent (`RandomAgent`)
- **Algorithm**: Random action selection
- **Purpose**: Baseline for performance comparison
- **File**: `ai/random_agent.py`

### 5. Human Agent (`HumanAgent`)
- **Algorithm**: Human input with optional AI assistance
- **Features**:
  - Interactive gameplay
  - Can receive recommendations from trained agents
  - Input validation and error handling
- **File**: `ai/human_agent.py`

## 📁 Project Structure

```
easy21/
├── ai/                          # Reinforcement learning agents
│   ├── __init__.py
│   ├── ai.py                    # Main AI framework and Learner class
│   ├── general_agent.py         # Base Agent class with ε-greedy policy
│   ├── monte_carlo.py           # Monte Carlo agent implementation
│   ├── sarsa_lambda.py          # Sarsa(λ) and Sarsa(λ) with approximation
│   ├── human_agent.py           # Human player agent
│   ├── random_agent.py          # Random baseline agent
│   └── results/                 # Generated plots and saved models
│       ├── *.png                # Value function visualizations
│       └── *.pickle             # Saved value functions and weights
├── game/                        # Game environment
│   ├── __init__.py
│   └── game.py                  # Easy21 game implementation
├── utils/                       # Utility functions
│   ├── __init__.py
│   └── display.py               # Plotting and visualization utilities
├── README.md                    # This file
└── Reference_Instructions.pdf   # Original assignment instructions
```

## 🚀 Usage

### Prerequisites
```bash
pip install numpy matplotlib
```

### Training Agents
```python
# Train Monte Carlo agent
from ai.ai import Learner
from ai.monte_carlo import MCAgent

mc_agent = MCAgent()
mc_learner = Learner(mc_agent)
mc_learner.learn(num_train_epochs=1000000, save=True)
```

### Playing the Game
```python
# Human vs AI
from game.game import Game
from ai.human_agent import HumanAgent
from ai.sarsa_lambda import SarsaAgent

# Load trained agent
sarsa_agent = SarsaAgent(_lambda=0.3)
sarsa_learner = Learner(sarsa_agent)
sarsa_learner.load_val_func()

# Play with AI assistance
human_agent = HumanAgent(assist_agent=sarsa_agent)
game = Game(ai=human_agent, stdout=True)
game.play_game()
```

### Running the Complete Experiment
```bash
cd ai
python ai.py
```

This will:
1. Train all agents (Monte Carlo, Sarsa, Sarsa with approximation)
2. Generate performance comparisons
3. Create value function visualizations
4. Plot MSE vs λ for different algorithms
5. Optionally allow human gameplay

## 📊 Results and Visualizations

The project generates several types of outputs:

### Value Function Plots
- 3D surface plots showing optimal action values
- Dealer showing card vs Player sum
- Generated for each agent type

### Performance Analysis
- Mean Squared Error vs λ parameter for Sarsa agents
- Learning curves showing convergence
- Performance comparisons between agents

### Saved Models
- Pickle files containing trained value functions
- Weight vectors for function approximation
- Can be loaded for evaluation or human gameplay

## 🔧 Key Features

- **Modular Design**: Clean separation between game environment and agents
- **Extensible Framework**: Easy to add new RL algorithms
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Interactive Play**: Human can play with AI assistance
- **Persistent Storage**: Save and load trained models
- **Visualization Tools**: 3D plots and learning curves

## 📚 Learning Objectives

This project demonstrates:
- **Monte Carlo Methods**: Learning from complete episodes
- **Temporal Difference Learning**: Online learning with bootstrapping
- **Eligibility Traces**: Efficient credit assignment in Sarsa(λ)
- **Function Approximation**: Handling large state spaces
- **Exploration vs Exploitation**: ε-greedy policies
- **Value Function Visualization**: Understanding learned policies

## 🤝 Contributing

Feel free to extend this project by:
- Implementing additional RL algorithms (Q-learning, DQN, etc.)
- Adding new game environments
- Improving visualization capabilities
- Optimizing performance

## 📄 License

This project is based on David Silver's Reinforcement Learning course materials. Please refer to the original course for licensing information. 


  
