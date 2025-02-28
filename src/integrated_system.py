
import os
import sys
from .wumpus_world_system import WumpusWorld

class IntegratedWumpusTicTacToe:
    """
    System that integrates the Wumpus World problem with Tic-Tac-Toe gameplay.
    Based on the Tic-Tac-Toe outcome, it decides whether to use the best move
    or a random move strategy for the Wumpus World.
    """
    
    def __init__(self, wumpus_size=4):
        """
        Initialize the integrated system.
        
        Args:
            wumpus_size (int): Size of the Wumpus World grid (N x N)
        """
        self.wumpus_world = WumpusWorld(wumpus_size)
        self.game_history = []
    
    def take_step_based_on_ttt_outcome(self, ttt_winner):
        """
        Take a step in the Wumpus World based on Tic-Tac-Toe outcome.
        
        Args:
            ttt_winner (int): 0 if LLM-1 wins (use best move), 1 if LLM-2 wins (use random move)
            
        Returns:
            dict: Status information about the step and current state
        """
        # Use Bayesian best move if LLM-1 wins (ttt_winner = 0)
        # Use random move if LLM-2 wins (ttt_winner = 1)
        use_best_move = (ttt_winner == 0)
        
        result = self.wumpus_world.take_step(use_best_move=use_best_move)
        
        # Record the step result
        step_record = {
            'step': self.wumpus_world.step,
            'ttt_winner': ttt_winner,
            'move_strategy': 'best' if use_best_move else 'random',
            'result': result
        }
        self.game_history.append(step_record)
        
        return result
    
    def get_current_state(self):
        """
        Get the current state of the Wumpus World.
        
        Returns:
            dict: Current game state
        """
        return self.wumpus_world.get_game_state()
    
    def get_current_risk_map(self):
        """
        Get the current risk map.
        
        Returns:
            numpy.ndarray: Risk matrix
        """
        return self.wumpus_world.get_current_risk()
    
    def is_game_over(self):
        """
        Check if the game is over.
        
        Returns:
            bool: True if game is over, False otherwise
        """
        return self.wumpus_world.is_game_over
    
    def found_gold(self):
        """
        Check if gold was found.
        
        Returns:
            bool: True if gold was found, False otherwise
        """
        return self.wumpus_world.found_gold
    
    def get_summary(self):
        """
        Get a summary of the game.
        
        Returns:
            dict: Game summary information
        """
        return {
            'total_steps': self.wumpus_world.step,
            'found_gold': self.wumpus_world.found_gold,
            'visited_cells': len(self.wumpus_world.visited),
            'total_cells': self.wumpus_world.size * self.wumpus_world.size,
            'best_move_count': sum(1 for item in self.game_history if item['move_strategy'] == 'best'),
            'random_move_count': sum(1 for item in self.game_history if item['move_strategy'] == 'random'),
            'deaths': sum(1 for item in self.game_history if item['result'].get('status') == 'died')
        }

# Example of how to use this in your main file:
if __name__ == "__main__":
    # This is just an example - you would replace this with your actual TTT system
    def get_ttt_outcome():
        """Simulate Tic-Tac-Toe game outcomes"""
        import random
        return random.randint(0, 1)  # 0 for LLM-1, 1 for LLM-2
    
    # Create the integrated system
    game = IntegratedWumpusTicTacToe(wumpus_size=6)
    
    # Run the game step by step based on TTT outcomes
    while not game.is_game_over():
        ttt_result = get_ttt_outcome()
        winner_name = "LLM-1" if ttt_result == 0 else "LLM-2"
        print(f"\nTic-Tac-Toe game result: {winner_name} wins")
        
        # Take a step in Wumpus World based on TTT outcome
        result = game.take_step_based_on_ttt_outcome(ttt_result)
        print(f"Move result: {result['message']}")
        
        if result['status'] == 'found_gold':
            print("🎉 Success! Gold found!")
            break
    
    # Game complete - show summary
    summary = game.get_summary()
    print("\nGame Summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
