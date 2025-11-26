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
        def generate_legal_moves(state: GameState):
            moves = []
            b = state.board
            allowed_squares = state.player_squares()

            for (i, j) in allowed_squares:
                if b.get((i, j)) != SudokuBoard.empty:
                    continue
                for value in range(1, N + 1):
                    if TabooMove((i, j), value) in state.taboo_moves:
                        continue
                    if not is_c0_ok(b, (i, j), value):
                        continue
                    moves.append(Move((i, j), value))
            return moves

        # helper fn; compute points gained by playing move on parent_state according to the scoring rules.
        def points_for_move(parent_state: GameState, move: Move) -> int:
            i, j = move.square
            parent_board = parent_state.board

            # A region is completed if all its cells are non-empty.
            # For this move, a region is newly completed if before the move
            # exactly one cell (this one) was empty.
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

            if completed == 0:
                return 0
            elif completed == 1:
                return 1
            elif completed == 2:
                return 3
            else:  # completed == 3
                return 7

        # helper fn; apply a move and return a new gamestate
        def apply_move(state: GameState, move: Move) -> GameState:
            new_state = copy.deepcopy(state)

            i, j = move.square
            value = move.value
            player = new_state.current_player  

            # board upate
            new_state.board.put((i, j), value)
            new_state.moves.append(move)

            # occupied squares update
            if player == 1:
                if new_state.occupied_squares1 is None:
                    new_state.occupied_squares1 = []
                if (i, j) not in new_state.occupied_squares1:
                    new_state.occupied_squares1.append((i, j))
            else:
                if new_state.occupied_squares2 is None:
                    new_state.occupied_squares2 = []
                if (i, j) not in new_state.occupied_squares2:
                    new_state.occupied_squares2.append((i, j))

            # scores update
            pts = points_for_move(state, move)
            if new_state.scores is None or len(new_state.scores) < 2:
                new_state.scores = [0, 0]
            new_state.scores[player - 1] += pts

            # switch player
            new_state.current_player = 2 if player == 1 else 1

            return new_state

        # helper fn; check if board is completely filled
        def board_is_full(state: GameState) -> bool:
            b = state.board
            for i in range(N):
                for j in range(N):
                    if b.get((i, j)) == SudokuBoard.empty:
                        return False
            return True

        # Evaluation function (heuristic)
        def score_state(for_state: GameState, depth: int) -> float:
            """
            Evaluation from the perspective of the ROOT player
            (game_state.current_player at the start of this turn).

            Components:
            - Strong weight on actual score difference.
            - Region control: favour regions where we have majority and few empties.
            - Mild positional bias (central control).
            - Small depth bonus for faster wins / slower losses.
            """
            root = root_player          
            my_idx = root - 1         
            opp_idx = 1 - my_idx

            b = for_state.board

            # 1. Actual score difference
            if for_state.scores is None or len(for_state.scores) < 2:
                my_score = 0
                opp_score = 0
            else:
                my_score = for_state.scores[my_idx]
                opp_score = for_state.scores[opp_idx]
            SCORE_DIFF_WEIGHT = 50.0
            score = SCORE_DIFF_WEIGHT * (my_score - opp_score)

            # ownership information
            pos_squares = for_state.occupied_squares1 if root == 1 else for_state.occupied_squares2
            neg_squares = for_state.occupied_squares2 if root == 1 else for_state.occupied_squares1

            if pos_squares is None:
                pos_squares = []
            if neg_squares is None:
                neg_squares = []

            owner = {}
            if for_state.occupied_squares1:
                for (i, j) in for_state.occupied_squares1:
                    owner[(i, j)] = 1
            if for_state.occupied_squares2:
                for (i, j) in for_state.occupied_squares2:
                    owner[(i, j)] = 2

            # 2. Region control & potential
            REGION_WEIGHT = 3.0

            def add_region_score(cells):
                nonlocal score
                empty = 0
                my_cells = 0
                opp_cells = 0

                for (i, j) in cells:
                    v = b.get((i, j))
                    if v == SudokuBoard.empty:
                        empty += 1
                    else:
                        p = owner.get((i, j), 0)
                        if p == root:
                            my_cells += 1
                        elif p != 0:
                            opp_cells += 1

                # completed regions are already accounted for in scores
                if empty == 0:
                    return

                if my_cells > opp_cells:
                    score += REGION_WEIGHT / empty
                elif opp_cells > my_cells:
                    score -= REGION_WEIGHT / empty

            # rows
            for i in range(N):
                add_region_score([(i, j) for j in range(N)])
            # columns
            for j in range(N):
                add_region_score([(i, j) for i in range(N)])
            # blocks
            for br in range(0, N, m):
                for bc in range(0, N, n):
                    cells = [
                        (i, j)
                        for i in range(br, br + m)
                        for j in range(bc, bc + n)
                    ]
                    add_region_score(cells)

            # 3. Positional bias: central control
            center = (N - 1) / 2.0
            POSITION_WEIGHT = 0.1

            for (i, j) in pos_squares:
                dx = abs(i - center)
                dy = abs(j - center)
                score += POSITION_WEIGHT * ((center - dx) + (center - dy))

            for (i, j) in neg_squares:
                dx = abs(i - center)
                dy = abs(j - center)
                score -= POSITION_WEIGHT * ((center - dx) + (center - dy))

            # 4. Depth bonus: prefer earlier advantages
            score += 0.01 * (10 - depth)

            return score

        # Alpha-beta search with skip-turn handling
        def alpha_beta(state: GameState,
                       depth: int,
                       alpha: float,
                       beta: float,
                       maximizing_player: bool,
                       max_depth: int) -> float:
            # terminal or depth limit
            if depth >= max_depth or board_is_full(state):
                return score_state(state, depth)

            moves = generate_legal_moves(state)

            # if no moves for this player: skip turn
            if not moves:
                # avoid infinite skipping; if also full, we treat as terminal
                if board_is_full(state):
                    return score_state(state, depth)
                
                state_after_skip = copy.deepcopy(state)
                state_after_skip.current_player = 2 if state.current_player == 1 else 1
                return alpha_beta(state_after_skip, depth + 1, alpha, beta,
                                  not maximizing_player, max_depth)

            if maximizing_player:
                value = float('-inf')
                for move in moves:
                    child = apply_move(state, move)
                    value = max(value, alpha_beta(child, depth + 1, alpha, beta,
                                                  False, max_depth))
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return value
            else:
                value = float('inf')
                for move in moves:
                    child = apply_move(state, move)
                    value = min(value, alpha_beta(child, depth + 1, alpha, beta,
                                                  True, max_depth))
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
                return value

        # Root; generate legal moves and run iterative deepening
        root_state = copy.deepcopy(game_state)
        root_moves = generate_legal_moves(root_state)

        if not root_moves:
            # no legal moves; nothing to propose; skip
            return

        # quick fallback; proposes a random legal move immediately
        fallback_move = random.choice(root_moves)
        self.propose_move(fallback_move)

        # iterative deepening; keeps improving best move as time allows
        depth_limit = 1
        while True:
            best_value = float('-inf')
            best_move = fallback_move

            for move in root_moves:
                child_state = apply_move(root_state, move)
                value = alpha_beta(child_state,
                                   depth=1,
                                   alpha=float('-inf'),
                                   beta=float('inf'),
                                   maximizing_player=False,
                                   max_depth=depth_limit)
                if value > best_value:
                    best_value = value
                    best_move = move

            self.propose_move(best_move)

            depth_limit += 1
            # no explicit time check: the framework will terminate this
            # process after the allowed time; last proposed move is used.
