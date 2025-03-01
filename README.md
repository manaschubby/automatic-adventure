# Automatic Adventure

A comprehensive AI system integrating Tic-Tac-Toe gameplay with Wumpus World pathfinding using LLM decision-making.

## Our Approach

### 1. Problem Decomposition
- Split the complex problem into three distinct tasks:
  1. LLM-powered Tic-Tac-Toe with statistical analysis
  2. Bayesian-based Wumpus World pathfinding
  3. Integration of both systems for dynamic strategy selection

### 2. LLM Decision Making (Task 1)
- **Multi-Provider Support**: Supports both Google Gemini and Groq for comparison
- **Statistical Rigor**: Implements Bernoulli trials for win/loss analysis
- **Key Features**:
  - Structured prompt engineering for consistent gameplay
  - Binomial distribution analysis of outcomes
  - Persistence of game states for long-running trials
  - Automated statistical visualization

### 3. Wumpus World Navigation (Task 2)
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

### 4. Integrated System (Task 3)
- **Dynamic Strategy Selection**:
  - Uses Tic-Tac-Toe outcomes to determine Wumpus World strategy
  - Winner-based decision making:
    * Player 1 wins → Use Bayesian best-move strategy
    * Player 2 wins → Use random exploration strategy
    * Draw → Default to best-move strategy
- **Adaptive Behavior**:
  - Real-time strategy adjustment based on game outcomes
  - Balance between exploration and exploitation
  - Learning from failures and successes

### 5. Technical Implementation
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
  - Comprehensive error reporting

### 6. Visualization and Analysis
- **Real-time Visualization**:
  - Risk map generation for each step
  - Binomial distribution plots
  - Visit frequency heatmaps
- **Results Analysis**:
  - Statistical analysis of game outcomes
  - Path efficiency metrics
  - Strategy effectiveness comparison

## Project Structure

```
automatic-adventure/
├── src/
│   ├── __main__.py           # Main CLI interface
│   ├── integrated_system.py  # Integration of Wumpus World with Tic-Tac-Toe
│   ├── llm.py               # LLM provider implementations (Gemini, Groq)
│   ├── tic_tac_toe.py       # Tic-Tac-Toe game implementation
│   ├── tic_tac_toe_solver.py # LLM-based Tic-Tac-Toe solver
│   ├── wumpus_world_system.py # Wumpus World implementation
│   ├── prompts/
│   │   └── tic_tac_toe_prompt.txt # Prompt template for Tic-Tac-Toe
│   └── utils/
│       ├── __init__.py
│       └── plot_binomial_distribution.py
```

## Requirements

- Python 3.10 or higher
- Google Gemini API key
- (Optional) Groq API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/automatic-adventure.git
cd automatic-adventure
```

2. Install dependencies:
```bash
pip install -r requirements.txt
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
  --no-prompt          Run all trials without prompting
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

## Key Components

### 1. LLM Integration (`src/llm.py`)

Provides a unified interface for different LLM providers:

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

- `wumpus_output/`: Contains risk map visualizations
  - `risk_step_XXX.png`: Risk map images for each step

## Benefits of Our Approach

1. **Robustness**:
   - Multiple fallback strategies
   - Error recovery mechanisms
   - State persistence

2. **Scalability**:
   - Support for different board sizes
   - Multiple LLM providers
   - Configurable parameters

3. **Analysis Capabilities**:
   - Statistical validation
   - Visual result presentation
   - Performance metrics

4. **Flexibility**:
   - Modular component design
   - Easy strategy modification
   - Configurable runtime behavior

