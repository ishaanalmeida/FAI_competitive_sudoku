"""
Debug: check если minimax вызывает exception
"""
import sys
import traceback
sys.path.append('/Users/none/Documents/github/FAI_competitive_sudoku/competitive_sudoku')

from competitive_sudoku.sudoku import GameState, SudokuBoard, Move
from team70_A1.sudokuai import SudokuAI

board = SudokuBoard(2, 2)
game_state = GameState(board=board, current_player=1)

ai = SudokuAI()

try:
    ai.compute_best_move(game_state)
except Exception as e:
    print(f"\n❌ EXCEPTION CAUGHT: {type(e).__name__}")
    print(f"Message: {e}")
    print("\nTraceback:")
    traceback.print_exc()
