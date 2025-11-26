#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

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
        N = game_state.board.N

        rw = game_state.board.region_width()
        rh = game_state.board.region_height()

        turn_idx = len(game_state.moves)

        heur_desired_x: float = N / 2.0
        heur_desired_y: int =  N - 1
        print(heur_desired_x, heur_desired_y)
        if game_state.current_player == 2:
            heur_desired_y = 0

        def score_move(i, j, val, for_state):
            def check_vertical():
                for n in range(0, j):
                    if for_state.board.get((i, n)) == SudokuBoard.empty:
                        return False
                for n in range(j + 1, N):
                    if for_state.board.get((i, n)) == SudokuBoard.empty:
                        return False
                return True
            def check_horizontal():
                for m in range(0, i):
                    if for_state.board.get((m, j)) == SudokuBoard.empty:
                        return False
                for m in range(i+1, N):
                    if for_state.board.get((m, j)) == SudokuBoard.empty:
                        return False
                return True
            def check_region():
                # check region
                region_i = int(i / rh)
                region_j = int(j / rw)
                for ri in range(region_i * rh, (region_i + 1) * rh):
                    for rj in range(region_j * rh, (region_j + 1) * rh):
                        if for_state.board.get((ri, rj)) == SudokuBoard.empty:
                            return False
                return True
            regions_completed = 0
            if check_horizontal():
                regions_completed += 1
            if check_vertical():
                regions_completed += 1
            if check_region():
                regions_completed += 1
            if regions_completed == 0:
                return 0
            elif regions_completed == 1:
                return 1
            elif regions_completed == 2:
                return 3
            elif regions_completed == 3:
                return 7
            else:
                raise Exception("completed > 3 regions in one go during scoring")

        def possible(i, j, value, for_state: GameState):
            # Basic checks
            if not (for_state.board.get((i, j)) == SudokuBoard.empty \
                   and not TabooMove((i, j), value) in for_state.taboo_moves):
                return False
            
            # Sudoku constraints
            board = for_state.board
            
            # Check row
            for c in range(N):
                if c != j and board.get((i, c)) == value:
                    return False
            
            # Check col
            for r in range(N):
                if r != i and board.get((r, j)) == value:
                    return False
            
            # Check region
            m = board.m
            n = board.n
            r_start = (i // m) * m
            c_start = (j // n) * n
            for r in range(r_start, r_start + m):
                for c in range(c_start, c_start + n):
                    if (r, c) != (i, j) and board.get((r, c)) == value:
                        return False
            
            return True

        def get_resulting_state(sq, val, from_state) -> GameState:
            """this is really inefficient but probably alright for now
            in the future we should not use copies of the GameState object at all"""
            new_state = copy.deepcopy(from_state)
            new_state.board.put(sq, val)
            new_state.moves.append(Move(square=sq, value=val))
            if from_state.current_player == 1:
                new_state.current_player = 2
                new_state.occupied_squares1.append(sq)
            else:
                new_state.current_player = 1
                new_state.occupied_squares2.append(sq)

            return new_state

        def score_state(for_state: GameState, depth: int) -> float:
            """simple inefficient scoring, slightly prefers centre X values and further y values"""
            score = 0
            if game_state.current_player == 1: # opposite of current player for scoring
                positive_squares = for_state.occupied_squares1
                negative_squares = for_state.occupied_squares2
            else:
                positive_squares = for_state.occupied_squares2
                negative_squares = for_state.occupied_squares1
            #print("SCORING")
            #print("  ", string_state(depth, for_state))

            # slightly score filling up a square
            regions = [
                [[0, 0] for _ in range(N // rw)]  # columns of regions
                for _ in range(N // rh)           # rows of regions
            ]

            HEUR_PREFER_FILLING_REGION = 0.1
            HEUR_PREFER_CENTRE_X = 0.0001
            HEUR_PREFER_FAR_Y = 0.001

            if len(positive_squares) > 0:
                # prefer further y squares
                pos_y_min = positive_squares[0][0]
                pos_y_max = positive_squares[0][0]
                for square in positive_squares:
                    pos_y_min = min(pos_y_min, square[0])
                    pos_y_max = max(pos_y_max, square[0])

                    # slightly prefer X centre
                    score += (N / (0.5 + abs(N*0.5 - square[1]))) * HEUR_PREFER_CENTRE_X

                    region_i = int(square[0] / rh)
                    region_j = int(square[1] / rw)
                    regions[region_i][region_j][0] += 1

                score += (pos_y_max - pos_y_min) * HEUR_PREFER_FAR_Y
            if len(negative_squares) > 0:
                # prefer further y squares
                neg_y_min = negative_squares[0][0]
                neg_y_max = negative_squares[0][0]
                for square in negative_squares:
                    neg_y_min = min(neg_y_min, square[0])
                    neg_y_max = max(neg_y_max, square[0])

                    # slightly prefer X centre
                    score -= (N / (0.5 + abs(N*0.5 - square[1]))) * HEUR_PREFER_CENTRE_X

                    region_i = int(square[0] / rh)
                    region_j = int(square[1] / rw)
                    regions[region_i][region_j][1] += 1

                score -= (neg_y_max - neg_y_min) * HEUR_PREFER_FAR_Y
            region_scoring = 0
            for region_i in range(N // rh):
                for region_j in range(N // rw):
                    region = regions[region_i][region_j]
                    our_numbers_in_region = region[0]
                    adversary_numbers_in_region = region[1]
                    # prefer multiple numbers in same region with no adversary numbers
                    # as adversary numbers prevent us from filling the square
                    if adversary_numbers_in_region == 0 and our_numbers_in_region > 1:
                        region_scoring += (our_numbers_in_region) * HEUR_PREFER_FILLING_REGION
                    if our_numbers_in_region == 0 and adversary_numbers_in_region > 1:
                        region_scoring -= (adversary_numbers_in_region) * HEUR_PREFER_FILLING_REGION
            score += region_scoring


            # print("  pos: ", positive_squares)
            # print("  neg: ", negative_squares)
            # print("    ->", score, (pos_y_min, pos_y_max), (neg_y_min, neg_y_max))

            for z in range(0, depth):
                d = len(for_state.moves) - depth + z
                last_move = for_state.moves[d]
                i, j = last_move.square
                # actual scoring of move
                move_scoring = score_move(i, j, last_move.value, for_state)

                if d % 2 == 0:
                    score += move_scoring
                    # print("   scoring ", last_move, " positively")
                else:
                    score -= move_scoring
                    # print("   scoring ", last_move, " negatively")
            return score

        def get_possible_moves(from_state: GameState):
            """optimized previous version; generate all legal moves for the current player from this state"""
            player_squares = from_state.player_squares()
            if player_squares is None:
                candidates = [(i, j) for i in range(N) for j in range(N)]
            else:
                candidates = player_squares

            return [
                Move((i, j), value)
                for (i, j) in candidates
                for value in range(1, N + 1)
                if possible(i, j, value, from_state)
            ]

        def string_state(depth, state):
            return f"State({depth} {[str(mov) for mov in state.moves[-depth:]]})"

        def rec_search_minimax(state, depth, alpha, beta, is_max_player):
            """Implementation based wikipedia pseudocode"""
            if depth == to_depth:
                return score_state(state, depth=depth), state

            # print("SEARCH, AT ", string_state(depth, state))

            node_children = []
            for move in get_possible_moves(from_state=state):
                new_state = get_resulting_state(move.square, move.value, from_state=state)
                node_children.append(new_state)

            if len(node_children) == 0:  # terminal state
                return score_state(state, depth=depth), state

            if is_max_player:
                value = float('-inf')
                chosen = None
                for child in node_children:
                    new_value, final_state = rec_search_minimax(child, depth + 1, alpha, beta, False)
                    if new_value > value:
                        value = new_value
                        chosen = final_state
                    if new_value >= beta:
                        break
                    alpha = max(alpha, new_value)
                return value, chosen
            else:
                value = float('inf')
                chosen = None
                for child in node_children:
                    new_value, final_state = rec_search_minimax(child, depth + 1, alpha, beta, True)
                    if new_value < value:
                        value = new_value
                        chosen = final_state
                    if new_value <= alpha:
                        break
                    beta = min(beta, new_value)
                return value, chosen
        # initial root state
        root = copy.deepcopy(game_state)
        turn_idx = len(game_state.moves)  # already defined earlier in your code

        # generate all legal root moves once
        legal_root_moves = get_possible_moves(root)
        if not legal_root_moves:
            # no legal moves -> let framework handle "cannot move" case
            return

        # quick fallback: propose a random legal move immediately
        fallback_move = random.choice(legal_root_moves)
        self.propose_move(fallback_move)

        print("BEGIN SEARCH")
        to_depth = 1
        while True:
            # use current to_depth as the global depth limit for rec_search_minimax
            value, best_state = rec_search_minimax(root, 0, float('-inf'), float('inf'), True)

            # best_state contains the full sequence of moves up to depth to_depth
            if best_state is not None and len(best_state.moves) > turn_idx:
                # the move at index turn_idx is the root move from this state
                best_root_move = best_state.moves[turn_idx]
                self.propose_move(best_root_move)
                # optional: debug
                # print(f"Depth {to_depth}, score {value}, best move {best_root_move}")

            to_depth += 1
            # no explicit time check: the outer framework will kill this process
            # after the allowed time; the last proposed move is used