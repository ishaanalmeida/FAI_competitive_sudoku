"""
Debug script to understand why AI misses obvious moves
"""
import sys
sys.path.append('/Users/none/Documents/github/FAI_competitive_sudoku/competitive_sudoku')

from competitive_sudoku.sudoku import GameState, SudokuBoard, Move, TabooMove
from team70_A1.sudokuai import SudokuAI

# Create test scenario: Row almost complete
board = SudokuBoard(2, 2)
board.put((0, 0), 1)
board.put((0, 1), 2)
board.put((0, 2), 3)
# (0, 3) needs value 4 to complete row

game_state = GameState(board=board, current_player=1)

print("Board state:")
print(f"Row 0: {[board.get((0, j)) for j in range(4)]}")
print(f"Empty squares: {[(i,j) for i in range(4) for j in range(4) if board.get((i,j)) == SudokuBoard.empty]}")
print()

# Create AI and manually check what moves it considers
ai = SudokuAI()

class DebugAI(SudokuAI):
    def propose_move(self, move):
        print(f"AI proposes: {move.square} -> {move.value}")
        super().propose_move(move)

debug_ai = DebugAI()

print("Running AI...")
try:
    debug_ai.compute_best_move(game_state)
except:
    pass

print("\nDone")
