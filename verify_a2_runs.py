import sys
import os
import copy
import multiprocessing
import time

# Add current dir to path to find competitive_sudoku package
# Add competitive_sudoku dir to path to find the inner competitive_sudoku package
sys.path.append('/Users/none/Documents/github/FAI_competitive_sudoku/competitive_sudoku')

from competitive_sudoku.sudoku import SudokuBoard, GameState
from team70_A2.sudokuai import SudokuAI

def test_a2():
    print("Initializing Board...")
    board = SudokuBoard(3, 3)
    gs = GameState(board, copy.deepcopy(board))
    
    player = SudokuAI()
    
    print("Computing best move (Iterative Deepening)...")
    
    # We need to use a shared object to see if it proposed anything if running in process.
    # But for a simple test, just seeing if it runs without crashing is 90%.
    
    p = multiprocessing.Process(target=player.compute_best_move, args=(gs,))
    p.start()
    time.sleep(2)
    if p.is_alive():
        print("Agent is running loop (good). Terminating.")
        p.terminate()
    else:
        print("Agent stopped unexpectedly (maybe crashed?). Check stderr.")
        
    print("Test Finished.")

if __name__ == "__main__":
    test_a2()
