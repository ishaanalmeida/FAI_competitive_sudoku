import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

import copy # this is part of the python standard library

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Anytime search: we repeatedly improve our choice for the current move
        using iterative-deepening alpha-beta. The framework will stop us after
        the allotted time; the last move proposed via self.propose_move is used.
        """
        board = game_state.board
        N = board.N
        m = board.m  # rows per block
        n = board.n  # cols per block
        root_player = game_state.current_player
        
        # Precompute block indices for faster lookup
        blocks = []
        for r in range(0, N, m):
            for c in range(0, N, n):
                blocks.append([(i, j) for i in range(r, r + m) for j in range(c, c + n)])

        # helper fn; C0 check (row, column, block uniqueness of non-empty)
        def is_c0_ok(b: SudokuBoard, square, value: int) -> bool:
            i, j = square
            # row
            for col in range(N):
                if col != j and b.get((i, col)) == value:
                    return False
            # column
            for row in range(N):
                if row != i and b.get((row, j)) == value:
                    return False
            # block
            block_row_start = (i // m) * m
            block_col_start = (j // n) * n
            for r in range(block_row_start, block_row_start + m):
                for c in range(block_col_start, block_col_start + n):
                    if (r, c) != (i, j) and b.get((r, c)) == value:
                        return False
            return True

        # helper fn; generate all legal moves for the current player
        # ENHANCEMENT: Prioritize "naked singles" and scoring moves
        def generate_legal_moves(state: GameState):
            moves = []
            b = state.board
            allowed_squares = state.player_squares()
            if allowed_squares is None:
                allowed_squares = [(r, c) for r in range(N) for c in range(N)]
            
            forced_moves = []
            scoring_moves = []
            normal_moves = []

            for (i, j) in allowed_squares:
                if b.get((i, j)) != SudokuBoard.empty:
                    continue
                
                valid_values = []
                for value in range(1, N + 1):
                    if TabooMove((i, j), value) in state.taboo_moves:
                        continue
                    if is_c0_ok(b, (i, j), value):
                        valid_values.append(value)
                
                # If only one valid value for this square, it is a "naked single" 
                # (constrained by Sudoku rules + Taboo). This is a very strong move.
                if len(valid_values) == 1:
                    move = Move((i, j), valid_values[0])
                    forced_moves.append(move)
                else:
                    for value in valid_values:
                        move = Move((i, j), value)
                        # Check if it creates points (completes a region)
                        if points_for_move(state, move) > 0:
                            scoring_moves.append(move)
                        else:
                            normal_moves.append(move)

            # HEURISTIC: If there are forced moves, ONLY explore those. 
            # This drastically reduces branching factor by effectively committing to "obvious" moves.
            if forced_moves:
                return forced_moves

            # Otherwise, prioritize scoring moves, then normal moves.
            # Shuffle normal moves to add variety if scores are equal (helps avoid loops/predictability)
            random.shuffle(normal_moves)
            return scoring_moves + normal_moves

        # helper fn; compute points gained by playing move on parent_state according to the scoring rules.
        def points_for_move(parent_state: GameState, move: Move) -> int:
            i, j = move.square
            parent_board = parent_state.board
            # Temporarily put the value to check completion (conceptually)
            # Actually, we check emptiness count without the move
            
            completed = 0
            
            # row
            empty_count = 0
            for col in range(N):
                if parent_board.get((i, col)) == SudokuBoard.empty:
                    empty_count += 1
            if empty_count == 1:
                completed += 1

            # column
            empty_count = 0
            for row in range(N):
                if parent_board.get((row, j)) == SudokuBoard.empty:
                    empty_count += 1
            if empty_count == 1:
                completed += 1

            # block
            block_row_start = (i // m) * m
            block_col_start = (j // n) * n
            empty_count = 0
            for r in range(block_row_start, block_row_start + m):
                for c in range(block_col_start, block_col_start + n):
                    if parent_board.get((r, c)) == SudokuBoard.empty:
                        empty_count += 1
            if empty_count == 1:
                completed += 1

            if completed == 0: return 0
            elif completed == 1: return 1
            elif completed == 2: return 3
            else: return 7

        # helper fn; apply a move and return a new gamestate
        def apply_move(state: GameState, move: Move) -> GameState:
            new_state = copy.deepcopy(state)
            i, j = move.square
            value = move.value
            player = new_state.current_player  

            new_state.board.put((i, j), value)
            new_state.moves.append(move)

            # update occupied lists
            if player == 1:
                if new_state.occupied_squares1 is None: new_state.occupied_squares1 = []
                if (i, j) not in new_state.occupied_squares1: new_state.occupied_squares1.append((i, j))
            else:
                if new_state.occupied_squares2 is None: new_state.occupied_squares2 = []
                if (i, j) not in new_state.occupied_squares2: new_state.occupied_squares2.append((i, j))

            pts = points_for_move(state, move)
            if new_state.scores is None or len(new_state.scores) < 2:
                new_state.scores = [0, 0]
            new_state.scores[player - 1] += pts

            new_state.current_player = 2 if player == 1 else 1
            return new_state

        def board_is_full(state: GameState) -> bool:
            # Optimization: check if number of occupied squares == N*N
            # But simpler to just scan or keep a counter. Scanning is O(N^2).
            # Given N=9, this is small.
            for i in range(N):
                for j in range(N):
                    if state.board.get((i, j)) == SudokuBoard.empty:
                        return False
            return True

        # Evaluation function (heuristic)
        def score_state(for_state: GameState, depth: int) -> float:
            root = root_player          
            my_idx = root - 1         
            opp_idx = 1 - my_idx

            if for_state.scores is None or len(for_state.scores) < 2:
                my_score = 0
                opp_score = 0
            else:
                my_score = for_state.scores[my_idx]
                opp_score = for_state.scores[opp_idx]
            
            # 1. Score Difference (Primary Objective)
            SCORE_DIFF_WEIGHT = 100.0 
            score = SCORE_DIFF_WEIGHT * (my_score - opp_score)

            # 2. Region Control (Heuristic)
            # Reward having majority in a region (row/col/block) that is close to completion
            # This encourages taking "ownership" of almost-full regions
            REGION_WEIGHT = 5.0
            
            # Helper to get owner map securely
            owner_map = {}
            p1_sq = for_state.occupied_squares1 or []
            p2_sq = for_state.occupied_squares2 or []
            for sq in p1_sq: owner_map[sq] = 1
            for sq in p2_sq: owner_map[sq] = 2
            
            def eval_region(cells):
                nonlocal score
                empty_count = 0
                my_count = 0
                opp_count = 0
                for (r, c) in cells:
                    val = for_state.board.get((r, c))
                    if val == SudokuBoard.empty:
                        empty_count += 1
                    else:
                        owner = owner_map.get((r, c), 0)
                        if owner == root:
                            my_count += 1
                        elif owner != 0:
                            opp_count += 1
                
                # If region is full, points are already awarded.
                # If region is empty, it's neutral.
                if 0 < empty_count < N:
                    # Heuristic: if we dominate a region, we are likely to get the last point
                    margin = my_count - opp_count
                    # Weight by how close it is to finishing (fewer empty -> more urgent/valuable)
                    score += (margin * REGION_WEIGHT) / (empty_count + 1)

            # Eval all regions
            for i in range(N): eval_region([(i, j) for j in range(N)]) # Rows
            for j in range(N): eval_region([(i, j) for i in range(N)]) # Cols
            for block_cells in blocks: eval_region(block_cells)        # Blocks
            
            # 3. Mobility (Heuristic)
            # Having more moves than opponent is generally good.
            # Calculating exact mobility is expensive, so we skip it to save time for deeper search.
            
            # 4. Center Control (Positional Heuristic)
            # Central squares often participate in more constraints
            center = (N - 1) / 2.0
            POS_WEIGHT = 0.5
            for (r, c) in p1_sq:
                dist = abs(r - center) + abs(c - center)
                if root == 1: score += POS_WEIGHT * (N - dist)
            for (r, c) in p2_sq:
                dist = abs(r - center) + abs(c - center)
                if root == 2: score += POS_WEIGHT * (N - dist)

            return score

        # Alpha-beta search
        def alpha_beta(state: GameState, depth: int, alpha: float, beta: float,
                       maximizing_player: bool, max_depth: int) -> float:
            if depth >= max_depth or board_is_full(state):
                return score_state(state, depth)

            moves = generate_legal_moves(state)

            if not moves:
                if board_is_full(state):
                    return score_state(state, depth)
                # Skip turn
                state_after_skip = copy.deepcopy(state)
                state_after_skip.current_player = 2 if state.current_player == 1 else 1
                return alpha_beta(state_after_skip, depth + 1, alpha, beta,
                                  not maximizing_player, max_depth)

            if maximizing_player:
                value = float('-inf')
                for move in moves:
                    child = apply_move(state, move)
                    value = max(value, alpha_beta(child, depth + 1, alpha, beta, False, max_depth))
                    alpha = max(alpha, value)
                    if alpha >= beta: break
                return value
            else:
                value = float('inf')
                for move in moves:
                    child = apply_move(state, move)
                    value = min(value, alpha_beta(child, depth + 1, alpha, beta, True, max_depth))
                    beta = min(beta, value)
                    if beta <= alpha: break
                return value

        # Root Search
        root_state = copy.deepcopy(game_state)
        legal_moves = generate_legal_moves(root_state)

        if not legal_moves:
            return

        # Initialize with random choice in case we timeout immediately
        best_move_so_far = random.choice(legal_moves)
        self.propose_move(best_move_so_far)

        # Iterative Deepening
        # Dynamic depth: if branching factor is small (forced moves), we might go very deep.
        # If branching is large, we stay shallow.
        depth_limit = 1
        while True:
            # We sort root moves by score for better alpha-beta ordering
            # (Though generate_legal_moves already puts scoring moves first, which helps)
            
            best_val = float('-inf')
            
            # Simple move ordering at root
            # Note: We can reuse the ordering from previous iteration if we stored it, 
            # but for now we just rely on generate_legal_moves heuristics.
            
            current_iter_best = None
            
            for move in legal_moves:
                child = apply_move(root_state, move)
                val = alpha_beta(child, 1, float('-inf'), float('inf'), False, depth_limit)
                
                if val > best_val:
                    best_val = val
                    current_iter_best = move
            
            if current_iter_best:
                best_move_so_far = current_iter_best
                self.propose_move(best_move_so_far)
            
            depth_limit += 1

