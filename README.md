# Automatic Adventure

A comprehensive AI system integrating Tic-Tac-Toe gameplay with Wumpus World pathfinding using LLM decision-making.

## Project Structure

```
CSF407_2025_2021B5A82926H/
├── src/
│   ├── __main__.py                         # Main CLI interface
│   ├── integrated_system.py                # Integration of Wumpus World with Tic-Tac-Toe
│   ├── llm.py                              # LLM provider implementations (Gemini, Groq)
│   ├── tic_tac_toe.py                      # Tic-Tac-Toe game implementation
│   ├── tic_tac_toe_solver.py               # LLM-based Tic-Tac-Toe solver
│   ├── wumpus_world_system.py              # Wumpus World implementation
│   ├── prompts/
│   │   └── tic_tac_toe_prompt.txt          # Prompt template for Tic-Tac-Toe
│   └── utils/
│       ├── __init__.py                     # Initialization file for utils module
│       └── plot_binomial_distribution.py
└── config.yml
```

## Requirements

- Python 3.10 or higher
- conda (Required)
- Google Gemini API key (Required)
- Groq API key (Optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ShaantanuJain/CSF407_2025_2021B5A82926H.git
cd CSF407_2025_2021B5A82926H
```

2. Create a conda environment:
```bash
conda env create -f config.yml
conda activate CSF407_2025_2021B5A82926H
```

## Usage

The system provides three main tasks that can be run through the CLI:

### Task 1: Tic-Tac-Toe Bernoulli Trials

Run multiple Tic-Tac-Toe games between two LLMs and analyze the results:

```bash
python -m src task1 [options]

Options:
  --gemini-key KEY     Specify Gemini API key
  --groq-key KEY       Specify Groq API key
  --continue           Continue from previous state
  --no-prompt          Run all trials without prompting (ideally choose this to let it run automatically)
```

Example:
```bash
python -m src task1 --gemini-key "your_key" --no-prompt
```

### Task 2: Wumpus World Simulation

Run the Wumpus World pathfinding simulation:

```bash
python -m src task2 [options]

Options:
  --size SIZE          Size of the Wumpus World grid
  --strategy STRATEGY  Movement strategy (best/random/mixed)
```

Example:
```bash
python -m src task2 --size 6 --strategy best
```

### Task 3: Integrated System

Run the integrated system where Tic-Tac-Toe outcomes determine Wumpus World movement strategies:

```bash
python -m src task3 [options]

Options:
  --gemini-key KEY     Specify Gemini API key
  --groq-key KEY       Specify Groq API key
  --size SIZE          Size of the Wumpus World grid
  --mode MODE          Run mode (auto/step)
```

Example:
```bash
python -m src task3 --gemini-key "your_key" --size 5 --mode step
```

## Key Components (Incase you want to use each component separately)

### 1. LLM Integration (`src/llm.py`)

Provides a unified interface for different LLM providers.  Factory design pattern is used to create and initialize providers. Helps in plug and play for different LLM providers:

```python
from src.llm import LLMFactory

# Create and initialize a provider
llm = LLMFactory.create_provider('gemini')
llm.initialize(api_key='YOUR_API_KEY')

# Generate text
response = llm.generate("Your prompt", temperature=0.7)
```

### 2. Tic-Tac-Toe System (`src/tic_tac_toe.py`, `src/tic_tac_toe_solver.py`)

Implements the Tic-Tac-Toe game logic and LLM-based solver:

```python
from src.tic_tac_toe_solver import play_game

# Play a game between two LLMs
winner = play_game(
    llm1_api_key="key1",
    llm2_api_key="key2",
    board_size=3,
    llm1_provider='gemini',
    llm2_provider='groq'
)
```

### 3. Wumpus World System (`src/wumpus_world_system.py`)

Implements the Wumpus World pathfinding logic:

```python
from src.wumpus_world_system import WumpusWorld

# Create and run a Wumpus World instance
world = WumpusWorld(size=4)
result = world.run_until_completion(strategy='best')
```

### 4. Integrated System (`src/integrated_system.py`)

Combines Tic-Tac-Toe and Wumpus World:

```python
from src.integrated_system import IntegratedWumpusTicTacToe

# Create and run the integrated system
game = IntegratedWumpusTicTacToe(wumpus_size=4)
result = game.take_step_based_on_ttt_outcome(ttt_winner=0)
```

## Output Files

The system generates various output files in two directories:

- `output/`: Contains game results and analysis
  - `tictactoe_results.json`: Tic-Tac-Toe game results
  - `wumpus_results.json`: Wumpus World simulation results
  - `integrated_results.json`: Integrated system results
  - `binomial_distribution.png`: Statistical analysis plot
  - `game_<timestamp>.json`: Output of every tic tac toe game played between LLMS

- `wumpus_output/`: Contains risk map visualizations
  - `risk_step_XXX.png`: Risk map images for each step



## Our Approach

### 1. LLM Decision Making (Task 1)
- **Multi-Provider Support**: Supports both Google Gemini and Groq for comparison
- **Statistical Rigor**: Implements Bernoulli trials for win/loss analysis
- **Key Features**:
  - Structured prompt engineering for consistent gameplay
  - Binomial distribution analysis of outcomes
  - Persistence of game states for long-running trials
  - Statistical visualization

### 2. Wumpus World Navigation (Task 2)
- **Bayesian Network Approach**:
  - Dynamic risk assessment using Bayesian inference
  - Probabilistic modeling of pit and wumpus locations
  - Real-time updating of beliefs based on percepts
- **Advanced Pathfinding**:
  - Multiple movement strategies (best, random, mixed)
  - Loop detection and avoidance
  - Death learning and risk adaptation
  - Visualization of risk maps
- **Knowledge Representation**:
  - Grid-based world model
  - Persistent memory of dangerous locations
  - Visit frequency tracking
  - Frontier-based exploration

### 3. Integrated System (Task 3)
- **Dynamic Strategy Selection**:
  - Uses Tic-Tac-Toe outcomes to determine Wumpus World strategy
  - Winner-based decision making:
    * Player 1 wins → Use Bayesian best-move strategy
    * Player 2 wins → Use random exploration strategy
    * Draw → Default to best-move strategy

### 4. Technical Implementation
- **Modular Architecture**:
  - Clear separation of concerns between components
  - Factory pattern for LLM provider management
  - Abstract base classes for extensibility
- **Data Management**:
  - JSON-based state persistence
  - Automated output organization
  - Visual analysis generation
- **Error Handling**:
  - Graceful degradation on API failures
  - State recovery mechanisms

### 5. Visualization and Analysis
- **Real-time Visualization**:
  - Risk map generation for each step
  - Binomial distribution plots
  - Visit frequency heatmaps
- **Results Analysis**:
  - Statistical analysis of game outcomes
  - Path efficiency metrics
  - Strategy effectiveness comparison
