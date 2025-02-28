from typing import List
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

def plot_binomial_distribution(outcomes: List[int], filename: str):
    """
    Create and save a binomial distribution plot of the outcomes
    """

    # Count wins for Player 1 (excluding draws)
    valid_games = [game for game in outcomes if game != 0]  # Exclude draws
    n_valid_games = len(valid_games)
    wins_p1 = valid_games.count(1)

    # Handle edge case with no valid games
    if n_valid_games == 0:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No valid games (no wins or losses, only draws)",
                 ha='center', va='center', fontsize=14)
        plt.title('Binomial Distribution of Tic Tac Toe Outcomes')
        plt.savefig(filename)
        plt.close()
        return

    # Calculate observed probability
    p_observed = wins_p1 / n_valid_games

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
