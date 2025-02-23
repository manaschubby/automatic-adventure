# automatic-adventure
LLM based Tic Tac Toe and Probabilistic Wumpus Solver

# Task Tracking

## Tic Tac Toe
- [x] Create a Tic Tac Toe Template
- [x] Store each game move
- [ ] Create a prompt for letting LLM take next Step in game based on current state and previous move
- [ ] Use LLM response to do next move
- [ ] Plotting response for 500 trials (Check how much this will cost)

## Wumpus (Make a better roadmap for this)
- [ ] Go through wumpus textbook problem and solutions described in Lecture slides
- [ ] Implement the wumpus solver bast on best move
- [ ] Implement based on random move

## Integrate both
- [ ] Let Wumpus solver call TicTacToe solver to give state 0 or 1
- [ ] Use tic tac toe solution to choose next move
- [ ] Visualize probabilities of risk on all moves and plot as color maps

## Reference file structure
```
├── src/
│   ├── prompts/
│   │   ├── tic_tac_toe_prompt.txt
│   ├── tic_tac_toe.py
│   ├── llm_caller.py
│   ├── wumpus_world_system.py
│   ├── integrated_system.py
│   ├── Exercise1.json
│   ├── Exercise1.png
│   ├── Exercise2_plots/
│   │   └── step_X_probabilities.png
├── README.md
├── config.yml
└── requirements.txt
```
