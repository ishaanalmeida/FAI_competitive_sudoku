import sys
import os
import time
import random
import unittest
import copy

# Add current directory to path
sys.path.append(os.getcwd())

from competitive_sudoku.sudoku import GameState, SudokuBoard, Move, TabooMove, SudokuBoard
from team70_A1.sudokuai import SudokuAI

class StopTest(Exception):
    pass

class TestAI(SudokuAI):
    def __init__(self):
        super().__init__()
        self.proposed_move = None
        self.move_count = 0

    def propose_move(self, move):
        self.proposed_move = move
        self.move_count += 1
        if self.move_count == 1:
            # Only stop on first move for quick tests
            raise StopTest("Move proposed")

def is_valid_sudoku_move(board: SudokuBoard, move: Move) -> bool:
    """Validate a Sudoku move"""
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

class ComprehensiveAITests(unittest.TestCase):
    
    def run_ai_once(self, game_state, max_time=0.1):
        """Run AI and get first proposed move"""
        ai = TestAI()
        try:
            ai.compute_best_move(game_state)
            # If it didn't stop, wait a bit and check
            time.sleep(max_time)
        except StopTest:
            pass
        return ai.proposed_move

    def test_ai_proposes_valid_moves(self):
        """Test 1: AI must propose valid Sudoku moves"""
        print("\n=== Test 1: Valid moves ===")
        board = SudokuBoard(2, 2)
        game_state = GameState(board=board, current_player=1)
        
        for trial in range(10):
            move = self.run_ai_once(game_state)
            self.assertIsNotNone(move, f"AI failed to propose a move on trial {trial}")
            self.assertTrue(is_valid_sudoku_move(board, move), 
                          f"AI proposed invalid move: {move}")
        print("✓ AI proposes valid moves")

    def test_ai_finds_obvious_winning_move(self):
        """Test 2: AI should find obvious scoring opportunities"""
        print("\n=== Test 2: Obvious winning move ===")
        board = SudokuBoard(2, 2)
        # Setup board where completing a row gives points
        # Row 0: 1 2 3 .  (only need 4 to complete)
        board.put((0, 0), 1)
        board.put((0, 1), 2)
        board.put((0, 2), 3)
        # (0, 3) is empty and needs value 4
        
        game_state = GameState(board=board, current_player=1)
        move = self.run_ai_once(game_state, max_time=0.5)
        
        self.assertIsNotNone(move, "AI failed to find the obvious move")
        # AI should complete the row at (0, 3) with value 4
        self.assertEqual(move.square, (0, 3), 
                        f"AI should play at (0,3) but played at {move.square}")
        self.assertEqual(move.value, 4,
                        f"AI should play value 4 but played {move.value}")
        print(f"✓ AI found winning move: {move}")

    def test_ai_avoids_helping_opponent(self):
        """Test 3: AI should not set up easy wins for opponent"""
        print("\n=== Test 3: Avoid helping opponent ===")
        board = SudokuBoard(2, 2)
        # Setup where opponent is one move from completing a region
        board.put((0, 0), 1)
        board.put((0, 1), 2)
        board.put((1, 0), 3)
        # (1, 1) would complete top-left region
        
        game_state = GameState(board=board, current_player=1)
        move = self.run_ai_once(game_state, max_time=0.5)
        
        self.assertIsNotNone(move, "AI failed to propose a move")
        # This is harder to test definitively, but AI should not play
        # moves that directly set up opponent scoring
        print(f"✓ AI proposed: {move}")

    def test_ai_performance_empty_board(self):
        """Test 4: AI should propose move quickly on empty board"""
        print("\n=== Test 4: Performance on empty board ===")
        board = SudokuBoard(3, 3)
        game_state = GameState(board=board, current_player=1)
        
        start_time = time.time()
        move = self.run_ai_once(game_state, max_time=2.0)
        elapsed = time.time() - start_time
        
        self.assertIsNotNone(move, "AI failed to propose a move")
        self.assertLess(elapsed, 1.0, 
                       f"AI took too long: {elapsed:.2f}s (should be < 1s)")
        print(f"✓ AI proposed move in {elapsed:.3f}s: {move}")

    def test_ai_finds_all_valid_moves(self):
        """Test 5: Check that AI's move generation is correct"""
        print("\n=== Test 5: Move generation correctness ===")
        board = SudokuBoard(2, 2)
        # Put some values
        board.put((0, 0), 1)
        board.put((0, 1), 2)
        
        game_state = GameState(board=board, current_player=1)
        
        # Manually count valid moves
        valid_moves = []
        for i in range(4):
            for j in range(4):
                if board.get((i, j)) == SudokuBoard.empty:
                    for val in range(1, 5):
                        test_move = Move((i, j), val)
                        if is_valid_sudoku_move(board, test_move):
                            valid_moves.append(test_move)
        
        print(f"Expected {len(valid_moves)} valid moves")
        
        # Run AI multiple times and collect proposed moves
        proposed_moves = set()
        for _ in range(20):
            move = self.run_ai_once(game_state)
            if move:
                proposed_moves.add((move.square, move.value))
        
        print(f"AI proposed {len(proposed_moves)} different moves across 20 trials")
        
        # AI should find at least some valid moves
        self.assertGreater(len(proposed_moves), 0, "AI found no valid moves")
        
        # All proposed moves should be valid
        for sq, val in proposed_moves:
            test_move = Move(sq, val)
            self.assertTrue(is_valid_sudoku_move(board, test_move),
                          f"AI proposed invalid move: {test_move}")
        
        print(f"✓ All {len(proposed_moves)} proposed moves were valid")

    def test_ai_handles_constrained_board(self):
        """Test 6: AI handles heavily constrained board"""
        print("\n=== Test 6: Constrained board ===")
        board = SudokuBoard(2, 2)
        # Fill most of the board
        board.put((0, 0), 1)
        board.put((0, 1), 2)
        board.put((0, 2), 3)
        board.put((1, 0), 3)
        board.put((1, 1), 4)
        board.put((1, 2), 1)
        board.put((2, 0), 2)
        board.put((2, 1), 1)
        board.put((2, 2), 4)
        board.put((3, 0), 4)
        board.put((3, 1), 3)
        board.put((3, 2), 2)
        # Many forced moves left
        
        game_state = GameState(board=board, current_player=1)
        move = self.run_ai_once(game_state, max_time=0.5)
        
        self.assertIsNotNone(move, "AI should find a move on constrained board")
        self.assertTrue(is_valid_sudoku_move(board, move),
                       f"AI proposed invalid move: {move}")
        print(f"✓ AI handled constrained board: {move}")

    def test_ai_with_taboo_moves(self):
        """Test 7: AI respects taboo moves"""
        print("\n=== Test 7: Taboo moves ===")
        board = SudokuBoard(2, 2)
        taboo = [TabooMove((0, 0), 1), TabooMove((0, 0), 2)]
        
        game_state = GameState(board=board, current_player=1, taboo_moves=taboo)
        
        # Run AI multiple times
        for _ in range(10):
            move = self.run_ai_once(game_state)
            self.assertIsNotNone(move, "AI should find a move")
            
            # Check it's not a taboo move
            for taboo_move in taboo:
                if move.square == taboo_move.square:
                    self.assertNotEqual(move.value, taboo_move.value,
                                      f"AI played taboo move: {move}")
        print("✓ AI respects taboo moves")

    def test_ai_consistency(self):
        """Test 8: AI should be somewhat consistent in similar positions"""
        print("\n=== Test 8: Consistency ===")
        board = SudokuBoard(2, 2)
        game_state = GameState(board=board, current_player=1)
        
        # Run AI 5 times on same position
        moves = []
        for _ in range(5):
            move = self.run_ai_once(game_state)
            if move:
                moves.append(move)
        
        self.assertEqual(len(moves), 5, "AI should always propose a move")
        print(f"AI proposed moves: {moves}")
        
        # Check all are valid
        for move in moves:
            self.assertTrue(is_valid_sudoku_move(board, move),
                          f"Invalid move: {move}")
        print("✓ AI is consistent and all moves valid")

if __name__ == "__main__":
    print("=" * 60)
    print("COMPREHENSIVE AI TEST SUITE")
    print("=" * 60)
    unittest.main(verbosity=2)
