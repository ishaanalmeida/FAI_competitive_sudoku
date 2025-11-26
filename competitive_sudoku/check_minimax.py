"""
Analyze why AI is still losing most games.
Let's check if minimax is actually running or if fallback dominates.
"""
import sys
import time
sys.path.append('/Users/none/Documents/github/FAI_competitive_sudoku/competitive_sudoku')

from competitive_sudoku.sudoku import GameState, SudokuBoard, Move
from team70_A1.sudokuai import SudokuAI

class InstrumentedAI(SudokuAI):
    def __init__(self):
        super().__init__()
        self.proposals = []
    
    def propose_move(self, move):
        self.proposals.append((time.time(), move))
        super().propose_move(move)

# Simple board
board = SudokuBoard(2, 2)
game_state = GameState(board=board, current_player=1)

ai = InstrumentedAI()

print("Running AI for 0.5 seconds...")
import threading

def run_ai():
    try:
        ai.compute_best_move(game_state)
    except:
        pass

thread = threading.Thread(target=run_ai, daemon=True)
start = time.time()
thread.start()
thread.join(timeout=0.5)
elapsed = time.time() - start

print(f"\nElapsed: {elapsed:.3f}s")
print(f"Number of proposals: {len(ai.proposals)}")
print("\nProposal timeline:")
for i, (t, move) in enumerate(ai.proposals):
    print(f"  {i}: t={t-start:.4f}s move={move.square}->{move.value}")

if len(ai.proposals) == 1:
    print("\n⚠️  WARNING: Only 1 proposal = minimax didn't complete, using greedy fallback only!")
elif len(ai.proposals) > 1:
    print(f"\n✓ Minimax made {len(ai.proposals)-1} improvements over fallback")
