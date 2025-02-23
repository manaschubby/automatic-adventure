import json
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from tqdm import tqdm
import os
import argparse
from .tic_tac_toe_solver import play_game

def load_previous_outcomes(filename: str) -> List[int]:
    """
    Load previously saved outcomes from a JSON file
    """
    try:
        with open(f"{filename}.json", 'r') as f:
            data = json.load(f)
            return data["outcomes"]
    except FileNotFoundError:
        return []

def run_trials(n_trials: int = 5, continue_previous: bool = False) -> List[int]:
    """
    Run n_trials of Tic Tac Toe games and return the outcomes
    """
    outcomes = []
    if continue_previous:
        outcomes = load_previous_outcomes("output/Exercise1")
        print(f"Loaded {len(outcomes)} previous outcomes")
        remaining_trials = n_trials - len(outcomes)
    else:
        remaining_trials = n_trials

    if remaining_trials <= 0:
        print("Already have enough trials, no need to run more")
        return outcomes

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Use tqdm for progress bar
    start_index = len(outcomes)
    for i in tqdm(range(remaining_trials), desc="Running games"):
        winner = play_game(api_key, api_key)
        print(f"Game {start_index + i + 1} winner: {winner}")
        outcomes.append(winner)

    return outcomes

def save_outcomes(outcomes: List[int], filename: str):
    """
    Save the outcomes to a file (both JSON and TXT formats)
    """
    # Save as JSON
    with open(f"{filename}.json", 'w') as f:
        json.dump({
            "total_games": len(outcomes),
            "player1_wins": outcomes.count(1),
            "player2_wins": outcomes.count(2),
            "draws": outcomes.count(0),
            "outcomes": outcomes
        }, f, indent=4)

    # Save as TXT
    with open(f"{filename}.txt", 'w') as f:
        f.write(f"Total games played: {len(outcomes)}\n")
        f.write(f"Player 1 wins: {outcomes.count(1)}\n")
        f.write(f"Player 2 wins: {outcomes.count(2)}\n")
        f.write(f"Draws: {outcomes.count(0)}\n")
        f.write("\nDetailed outcomes:\n")
        f.write(str(outcomes))

def plot_binomial_distribution(outcomes: List[int], filename: str):
    """
    Create and save a binomial distribution plot of the outcomes
    """
    # Count wins for Player 1 (excluding draws)
    valid_games = [game for game in outcomes if game != 0]  # Exclude draws
    n_valid_games = len(valid_games)
    wins_p1 = valid_games.count(1)

    # Calculate observed probability
    p_observed = wins_p1 / n_valid_games if n_valid_games > 0 else 0

    # Create binomial distribution
    k = np.arange(0, n_valid_games + 1)
    binomial = binom.pmf(k, n_valid_games, 0.5)  # Expected probability is 0.5

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(k, binomial, alpha=0.5, color='blue', label='Expected Distribution')
    plt.axvline(x=wins_p1, color='red', linestyle='--',
                label=f'Observed wins (Player 1): {wins_p1}')

    plt.title('Binomial Distribution of Tic Tac Toe Outcomes')
    plt.xlabel('Number of Player 1 Wins')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add text box with statistics
    stats_text = f'Total valid games: {n_valid_games}\n'
    stats_text += f'Player 1 wins: {wins_p1}\n'
    stats_text += f'Observed probability: {p_observed:.3f}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run Tic Tac Toe trials')
    parser.add_argument('--continue', dest='continue_previous', action='store_true',
                      help='Continue from previous results')
    parser.add_argument('--trials', type=int, default=500,
                      help='Total number of trials to run (default: 500)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Run the trials
    print(f"Starting/Continuing trials until we have {args.trials} total...")
    outcomes = run_trials(args.trials, args.continue_previous)

    # Save outcomes
    save_outcomes(outcomes, "output/Exercise1")

    # Create and save plot
    plot_binomial_distribution(outcomes, "output/Exercise1.png")

    print("Analysis complete! Results saved in output directory.")
    print(f"Player 1 wins: {outcomes.count(1)}")
    print(f"Player 2 wins: {outcomes.count(2)}")
    print(f"Draws: {outcomes.count(0)}")

if __name__ == "__main__":
    main()
