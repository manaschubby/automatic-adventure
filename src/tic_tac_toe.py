import json
from typing import Tuple, Optional
from datetime import datetime

class TicTacToeGame:
    def __init__(self, n: int, game_id: str):
        self.n = n
        self.game_id = game_id
        self.board = [['' for _ in range(n)] for _ in range(n)]
        self.moves_history = []
        self.game_finished = False
        self.winner: str | None = None

    def save_to_file(self):
        """Saves the current game state to a JSON file"""
        game_data = {
            'game_id': self.game_id,
            'n': self.n,
            'board': self.board,
            'moves_history': self.moves_history,
            'game_finished': self.game_finished,
            'winner': self.winner
        }

        with open(f'output/game_{self.game_id}.json', 'w') as f:
            json.dump(game_data, f)

    @staticmethod
    def load_from_file(game_id: str) -> 'TicTacToeGame':
        """Loads a game from a JSON file"""
        with open(f'output/game_{game_id}.json', 'r') as f:
            game_data = json.load(f)

        game = TicTacToeGame(game_data['n'], game_data['game_id'])
        game.board = game_data['board']
        game.moves_history = game_data['moves_history']
        game.game_finished = game_data['game_finished']
        game.winner = game_data['winner']
        return game

    def check_winner(self) -> Optional[str]:
        # Check rows
        for row in self.board:
            if len(set(row)) == 1 and row[0] != '':
                return row[0]

        # Check columns
        for col in range(self.n):
            if len(set(row[col] for row in self.board)) == 1 and self.board[0][col] != '':
                return self.board[0][col]

        # Check diagonals
        main_diag = [self.board[i][i] for i in range(self.n)]
        if len(set(main_diag)) == 1 and main_diag[0] != '':
            return main_diag[0]

        anti_diag = [self.board[i][self.n-1-i] for i in range(self.n)]
        if len(set(anti_diag)) == 1 and anti_diag[0] != '':
            return anti_diag[0]

        return None

    def is_board_full(self) -> bool:
        return all(cell != '' for row in self.board for cell in row)

def create_game(n: int) -> str:
    """Creates a new game and returns the game ID"""
    game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    game = TicTacToeGame(n, game_id)
    game.save_to_file()
    return game_id

def play_move(game_id: str, player: int, position: Tuple[int, int]) -> Tuple[bool, Optional[int]]:
        """
        Makes a move in the specified game
        Returns (game_finished, winner_player_number)
        """
        # Load the game
        game = TicTacToeGame.load_from_file(game_id)

        # Validate move
        row, col = position
        if row < 0 or row >= game.n or col < 0 or col >= game.n:
            raise ValueError("Invalid position")
        if game.board[row][col] != '':
            raise ValueError("Position already taken")
        if game.game_finished:
            raise ValueError("Game is already finished")
        if player not in [1, 2]:
            raise ValueError("Invalid player number")

        # Determine move symbol based on player number
        move = 'X' if player == 1 else 'O'

        # Make the move
        game.board[row][col] = move
        game.moves_history.append({
            'player': player,
            'position': position,
            'move': move
        })

        # Check for winner or draw
        winner_symbol = game.check_winner()
        winner_player = None
        if winner_symbol:
            winner_player = 1 if winner_symbol == 'X' else 2

        if winner_symbol or game.is_board_full():
            game.game_finished = True
            game.winner = winner_symbol

        # Save the updated game state
        game.save_to_file()

        return game.game_finished, winner_player

# Example usage:
if __name__ == "__main__":
    # Create a new 3x3 game
    game_id = create_game(3)
    print(f"Created game with ID: {game_id}")

    try:
        # Player 1 makes a move (X)
        finished, winner = play_move(game_id, 1, (0, 0))
        print(f"Move 1 - Finished: {finished}, Winner: {winner}")

        # Player 2 makes a move (O)
        finished, winner = play_move(game_id, 2, (1, 0))
        print(f"Move 2 - Finished: {finished}, Winner: {winner}")

        # Player 1 makes a move (X)
        finished, winner = play_move(game_id, 1, (1, 1))
        print(f"Move 3 - Finished: {finished}, Winner: {winner}")

        # Player 2 makes a move (O)
        finished, winner = play_move(game_id, 2, (2, 0))
        print(f"Move 4 - Finished: {finished}, Winner: {winner}")

        # Player 1 makes a move (X)
        finished, winner = play_move(game_id, 1, (2, 2))
        print(f"Move 5 - Finished: {finished}, Winner: {winner}")

        # Load and print the game state
        game = TicTacToeGame.load_from_file(game_id)
        print("\nCurrent board:")
        for row in game.board:
            print(row)

        print("\nMoves history:")
        print(game.moves_history)

    except ValueError as e:
        print(f"Error: {e}")
