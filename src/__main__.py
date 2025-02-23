
from .tic_tac_toe_solver import play_game

winner = play_game("AIzaSyBvYBIQ4LpLlbv9tgM6lyUYQlkhF3NOTuc", "AIzaSyBvYBIQ4LpLlbv9tgM6lyUYQlkhF3NOTuc", board_size=9)
print(f"Player {winner} wins!" if winner != 0 else "It's a draw!")
