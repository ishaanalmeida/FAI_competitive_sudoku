import sys
import os
import time
import random
import unittest
import copy

# Add current directory to path
sys.path.append(os.getcwd())

from competitive_sudoku.sudoku import GameState, SudokuBoard, Move, TabooMove
# CHANGED: Import from team70_A1
from team70_A1.sudokuai import SudokuAI

class StopTest(Exception):
    pass

class TestAI(SudokuAI):
    def __init__(self):
        super().__init__()
        self.proposed_move = None

    def propose_move(self, move):
        self.proposed_move = move
        raise StopTest("Move proposed")

def is_valid_sudoku_move(board: SudokuBoard, move: Move) -> bool:
    row, col = move.square
    value = move.value
    
    if board.get(move.square) != SudokuBoard.empty:
        return False
        
    # Check row
    for c in range(board.N):
        if c != col and board.get((row, c)) == value:
            return False
            
    # Check col
    for r in range(board.N):
        if r != row and board.get((r, col)) == value:
            return False
            
    # Check region
    m = board.m
    n = board.n
    r_start = (row // m) * m
    c_start = (col // n) * n
    
    for r in range(r_start, r_start + m):
        for c in range(c_start, c_start + n):
            if (r, c) != (row, col) and board.get((r, c)) == value:
                return False
                
    return True

class AdvancedSudokuTests(unittest.TestCase):
    
    def run_ai_once(self, game_state):
        ai = TestAI()
        try:
            ai.compute_best_move(game_state)
        except StopTest:
            pass
        except Exception as e:
            # If AI returns without proposing (e.g. no moves), that's fine for some tests
            pass
        return ai.proposed_move

    def test_basic_constraints(self):
        """Test that AI respects basic row/col/region constraints on a 4x4 board."""
        board = SudokuBoard(2, 2)
        # Row 0: 1 . . 2
        board.put((0, 0), 1)
        board.put((0, 3), 2)
        # Row 1: . 3 . .
        board.put((1, 1), 3)
        
        allowed = []
        for r in range(4):
            for c in range(4):
                if board.get((r, c)) == SudokuBoard.empty:
                    allowed.append((r, c))
        
        game_state = GameState(board=board, allowed_squares1=allowed, occupied_squares1=[], current_player=1)
        
        for _ in range(20):
            move = self.run_ai_once(game_state)
            if move:
                self.assertTrue(is_valid_sudoku_move(board, move), f"AI proposed illegal move: {move}")

    def test_allowed_squares_restriction(self):
        """Test that AI only plays in allowed squares."""
        board = SudokuBoard(2, 2)
        allowed = [(3, 3)]
        game_state = GameState(board=board, allowed_squares1=allowed, occupied_squares1=[], current_player=1)
        
        for _ in range(10):
            move = self.run_ai_once(game_state)
            if move:
                self.assertEqual(move.square, (3, 3), "AI played outside allowed squares")

    def test_taboo_moves(self):
        """Test that AI avoids taboo moves."""
        board = SudokuBoard(2, 2)
        taboo = [TabooMove((0, 0), 1)]
        allowed = [(0, 0)]
        
        game_state = GameState(board=board, allowed_squares1=allowed, occupied_squares1=[], taboo_moves=taboo, current_player=1)
        
        for _ in range(10):
            move = self.run_ai_once(game_state)
            if move:
                self.assertNotEqual(move.value, 1, "AI played a taboo move")

    def test_standard_9x9_board(self):
        """Test on a standard 9x9 board."""
        board = SudokuBoard(3, 3)
        for i in range(8):
            board.put((0, i), i + 1)
            
        allowed = []
        for r in range(9):
            for c in range(9):
                if board.get((r, c)) == SudokuBoard.empty:
                    allowed.append((r, c))
                    
        game_state = GameState(board=board, allowed_squares1=allowed, occupied_squares1=[], current_player=1)
        
        for _ in range(10):
            move = self.run_ai_once(game_state)
            if move:
                self.assertTrue(is_valid_sudoku_move(board, move), f"AI proposed illegal move on 9x9: {move}")
                if move.square == (0, 8):
                    self.assertEqual(move.value, 9, "Only 9 should be allowed at (0,8)")

    def test_full_game_simulation(self):
        """Simulate a game loop on a 4x4 board to check for illegal moves over time."""
        print("\nRunning full game simulation...")
        board = SudokuBoard(2, 2)
        game_state = GameState(board=board, current_player=1)
        
        # We need to manually manage allowed squares if we want to simulate properly, 
        # but for this test we can just say all empty squares are allowed for simplicity
        # or use the game logic. Let's just update board.
        
        moves_made = 0
        max_moves = 16
        
        while moves_made < max_moves:
            # Update allowed squares based on empty spots
            allowed = []
            for r in range(4):
                for c in range(4):
                    if board.get((r, c)) == SudokuBoard.empty:
                        allowed.append((r, c))
            
            if not allowed:
                break
                
            game_state.allowed_squares1 = allowed
            game_state.allowed_squares2 = allowed # Simplified
            
            move = self.run_ai_once(game_state)
            
            if not move:
                # AI might not find a move if board is unsolvable or stuck, which is fine for this test
                # as long as it didn't crash or propose illegal.
                print("AI could not find a move (stuck or finished).")
                break
                
            self.assertTrue(is_valid_sudoku_move(board, move), f"AI proposed illegal move during simulation: {move}")
            
            # Apply move
            board.put(move.square, move.value)
            game_state.moves.append(move)
            moves_made += 1
            
        print(f"Simulation finished after {moves_made} moves.")

    def test_single_valid_move(self):
        """Test a scenario where only one specific move is valid."""
        board = SudokuBoard(2, 2)
        # Fill board such that only (0,0) -> 4 is valid
        # Row 0: . 1 2 3
        board.put((0, 1), 1)
        board.put((0, 2), 2)
        board.put((0, 3), 3)
        # Col 0 needs to be constrained too?
        # Let's just make it simple:
        # Row 0: . 1 2 3
        # Row 1: 3 2 1 4
        # Row 2: 1 3 4 2
        # Row 3: 2 4 3 1
        # The only empty spot is (0,0). 
        # Col 0 has 3, 1, 2. So 4 is missing.
        # Region 0 has ., 1, 3, 2. So 4 is missing.
        # Row 0 has ., 1, 2, 3. So 4 is missing.
        
        board.put((1, 0), 3); board.put((1, 1), 2); board.put((1, 2), 1); board.put((1, 3), 4)
        board.put((2, 0), 1); board.put((2, 1), 3); board.put((2, 2), 4); board.put((2, 3), 2)
        board.put((3, 0), 2); board.put((3, 1), 4); board.put((3, 2), 3); board.put((3, 3), 1)
        
        allowed = [(0, 0)]
        game_state = GameState(board=board, allowed_squares1=allowed, occupied_squares1=[], current_player=1)
        
        move = self.run_ai_once(game_state)
        if move:
            self.assertEqual(move.square, (0, 0))
            self.assertEqual(move.value, 4, f"AI should have found the only valid move 4, but proposed {move.value}")
        else:
            self.fail("AI failed to find the single valid move")

if __name__ == "__main__":
    unittest.main()
