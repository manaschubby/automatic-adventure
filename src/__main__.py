
import json
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from time import sleep
import os
from .tic_tac_toe_solver import play_game

def run_trials(n_trials: int = 5) -> List[int]:
    """
    Run n_trials of Tic Tac Toe games and return the outcomes
    """
    outcomes = []
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Use tqdm for progress bar
    for i in tqdm(range(n_trials), desc="Running games"):
        winner = play_game(api_key, api_key)
        print("Game", i+1, "winner:", winner, "Waiting 10s before next game")
        sleep(10)
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
    binomial = plt.binompmf(k, n_valid_games, 0.5)  # Expected probability is 0.5

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
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Run the trials
    print("Starting 5 trials of Tic Tac Toe...")
    outcomes = run_trials(5)

    # Save outcomes
    save_outcomes(outcomes, "output/Exercise1")

    # Create and save plot
    plot_binomial_distribution(outcomes, "output/Exercise1.png")

    print("Analysis complete! Results saved in output directory.")
    print(f"Player 1 wins: {outcomes.count(1)}")
    print(f"Player 2 wins: {outcomes.count(2)}")
    print(f"Draws: {outcomes.count(0)}")

main()
