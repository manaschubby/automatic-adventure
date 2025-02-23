from typing import Tuple
import json
from pathlib import Path
import os
from .llm import LLMProvider, LLMFactory
from .tic_tac_toe import create_game, play_move, TicTacToeGame

class TicTacToeSolver:
    def __init__(self, llm1: LLMProvider, llm2: LLMProvider):
        self.llm1 = llm1  # Player 1 (X)
        self.llm2 = llm2  # Player 2 (O)
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load the prompt template from the prompts directory"""
        prompt_path = Path(__file__).parent / "prompts" / "tic_tac_toe_prompt.txt"
        with open(prompt_path, 'r') as f:
            return f.read()

    def _format_board_for_prompt(self, game: TicTacToeGame) -> str:
        """Format the board as a string for the prompt"""
        board_str = ""
        for row in game.board:
            board_str += "|" + "|".join(cell if cell != "" else " " for cell in row) + "|\n"
        return board_str

    def _get_previous_move(self, game: TicTacToeGame) -> str:
        """Get the previous move in human-readable format"""
        if not game.moves_history:
            return "No previous moves"
        last_move = game.moves_history[-1]
        return f"row {last_move['position'][0]}, column {last_move['position'][1]}"


    def _prepare_prompt(self, game: TicTacToeGame, current_player: int) -> str:
            """Prepare the prompt for the LLM"""
            symbol = 'X' if current_player == 1 else 'O'
            opponent_symbol = 'O' if current_player == 1 else 'X'

            print(f"Board State for Player {current_player} ({symbol}):")
            print(self._format_board_for_prompt(game))

            return self.prompt_template.format(
                symbol=symbol,
                opponent_symbol=opponent_symbol,
                rows=game.n,
                layout=self._format_board_for_prompt(game),
                previous_move=self._get_previous_move(game)
            )

    def _get_move_from_llm(self, llm: LLMProvider, game: TicTacToeGame, current_player: int) -> Tuple[int, int]:
        """Get the next move from the LLM"""
        prompt = self._prepare_prompt(game, current_player)
        response = llm.generate(prompt, temperature=0.1)  # Low temperature for more consistent moves

        try:
            move_data = json.loads(response.text)
            return (move_data["move"]["row"], move_data["move"]["col"])
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid response from LLM: {response.text}") from e

    def play_game(self, board_size: int = 3) -> int:
        """
        Play a game of Tic Tac Toe between two LLMs.
        Returns:
            int: The winning player number (1 or 2) or 0 for a draw
        """
        game_id = create_game(board_size)
        game_finished = False
        winner = None
        current_player = 1

        while not game_finished:
            try:
                # Get the current LLM
                current_llm = self.llm1 if current_player == 1 else self.llm2

                # Load current game state
                game = TicTacToeGame.load_from_file(game_id)

                # Get move from current LLM
                move = self._get_move_from_llm(current_llm, game, current_player)
                print(f"Player {current_player} ({('X' if current_player == 1 else 'O')}) moves to {move}")

                # Play the move
                game_finished, winner = play_move(game_id, current_player, move)

                # Switch players
                current_player = 3 - current_player  # Switches between 1 and 2

            except ValueError as e:
                print(f"Error during move: {e}")
                # If there's an error, the current player forfeits
                return 3 - current_player

        return winner if winner is not None else 0

def play_game(llm1_api_key: str, llm2_api_key: str, board_size: int = 3) -> int:
    """
    High-level function to play a game of Tic Tac Toe between two LLMs.

    Args:
        llm1_api_key: API key for the first LLM
        llm2_api_key: API key for the second LLM
        board_size: Size of the Tic Tac Toe board (default 3x3)

    Returns:
        int: The winning player number (1 or 2) or 0 for a draw
    """
    # Create and initialize LLM providers
    llm1 = LLMFactory.create_provider('gemini')
    llm2 = LLMFactory.create_provider('gemini')

    llm1.initialize(api_key=llm1_api_key)
    llm2.initialize(api_key=llm2_api_key)

    # Create solver and play game
    solver = TicTacToeSolver(llm1, llm2)
    return solver.play_game(board_size)

# Example usage:
if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    api_key = api_key if api_key is not None else ""
    winner = play_game(api_key, api_key)
    print(f"Player {winner} wins!" if winner != 0 else "It's a draw!")
