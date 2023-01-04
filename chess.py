import math
import random
import time
import numpy as np
import itertools

SEED = 1
random.seed(SEED)

KING_NAME = 'king'
QUEEN_NAME = 'queen'
KNIGHT_NAMES = ['knight_1', 'knight_2']
ROOK_NAMES = ['rook_1', 'rook_2']
BISHOP_NAMES = ['bishop_1', 'bishop_2']
PAWN_NAMES = ['pawn_1', 'pawn_2', 'pawn_3', 'pawn_4', 'pawn_5', 'pawn_6', 'pawn_7', 'pawn_8']

# starting position using x-y coordinates with x, y in [1,8]
# white and black are mirrored, e.g. bishop_1 is on x=3 for both teams (they face each other at start)
STARTING_POSITION = {
    ('king', 'white'): np.array([5, 1]),
    ('queen', 'white'): np.array([4, 1]),
    ('knight_1', 'white'): np.array([2, 1]),
    ('knight_2', 'white'): np.array([7, 1]),
    ('bishop_1', 'white'): np.array([3, 1]),
    ('bishop_2', 'white'): np.array([6, 1]),
    ('rook_1', 'white'): np.array([1, 1]),
    ('rook_2', 'white'): np.array([8, 1]),
    ('pawn_1', 'white'): np.array([1, 2]),
    ('pawn_2', 'white'): np.array([2, 2]),
    ('pawn_3', 'white'): np.array([3, 2]),
    ('pawn_4', 'white'): np.array([4, 2]),
    ('pawn_5', 'white'): np.array([5, 2]),
    ('pawn_6', 'white'): np.array([6, 2]),
    ('pawn_7', 'white'): np.array([7, 2]),
    ('pawn_8', 'white'): np.array([8, 2]),
    ('queen_new', 'white'): np.array([0, 0]),
    ('knight_new', 'white'): np.array([0, 0]),
    ('bishop_new', 'white'): np.array([0, 0]),
    ('rook_new', 'white'): np.array([0, 0]),

    ('king', 'black'): np.array([5, 8]),
    ('queen', 'black'): np.array([4, 8]),
    ('knight_1', 'black'): np.array([2, 8]),
    ('knight_2', 'black'): np.array([7, 8]),
    ('bishop_1', 'black'): np.array([3, 8]),
    ('bishop_2', 'black'): np.array([6, 8]),
    ('rook_1', 'black'): np.array([1, 8]),
    ('rook_2', 'black'): np.array([8, 8]),
    ('pawn_1', 'black'): np.array([1, 7]),
    ('pawn_2', 'black'): np.array([2, 7]),
    ('pawn_3', 'black'): np.array([3, 7]),
    ('pawn_4', 'black'): np.array([4, 7]),
    ('pawn_5', 'black'): np.array([5, 7]),
    ('pawn_6', 'black'): np.array([6, 7]),
    ('pawn_7', 'black'): np.array([7, 7]),
    ('pawn_8', 'black'): np.array([8, 7]),
    ('queen_new', 'black'): np.array([0, 0]),
    ('knight_new', 'black'): np.array([0, 0]),
    ('bishop_new', 'black'): np.array([0, 0]),
    ('rook_new', 'black'): np.array([0, 0])
}


class Piece:
    def __init__(self, name, color):
        self.name = name
        self.piece_type = self.name.split(sep="_")[0]
        self.color = color
        self.starting_position = STARTING_POSITION[(name, color)]
        self.position = STARTING_POSITION[(name, color)]
        self.board = None
        self.possible_step_directions = None

    def __repr__(self):
        return f"{self.color[0]}_{self.name[0]}{self.name[-1]}"

    def __str__(self):
        return f"{self.color[0]}_{self.name[0]}{self.name[-1]}"

    def move(self, to):
        """to: new position (x,y)"""
        self.position = to

    def promote(self, new_position, verbose):
        pass

    def choose_new_position(self):
        """handles all new-position-selecting procedures the returned value can be fed to self.move()"""
        available_new_positions, _ = self.get_available_positions()
        opponent = self.board.player_1 if self.color != self.board.player_1.color else self.board.player_2
        for opp_piece in opponent.pieces:
            if self.blocking_check(from_piece=opp_piece):
                if self.remove_piece_ability(opp_piece):
                    # empty the list since the other positions are not relevant anymore
                    available_new_positions = []
                    available_new_positions.append(opp_piece.position)
        if available_new_positions:
            return random.choice(available_new_positions)
        else:
            return []

    def blocking_check(self, from_piece):
        if len(from_piece.get_opponent_pieces_blocking_opponent_king()) == 1:
            if self in from_piece.get_opponent_pieces_blocking_opponent_king():
                return True
            else:
                return False
        else:
            return False

    def get_positions_in_check(self):
        opponent = self.board.player_1 if self.color != self.board.player_1.color else self.board.player_2
        positions_in_check = []
        for p in opponent.pieces:
            positions_in_check.extend(p.positions_kept_in_check())
        return positions_in_check

    def positions_kept_in_check(self):
        check_threat = []

        # if king
        if self.piece_type == KING_NAME:
            for position in self.board.all_positions:
                if np.linalg.norm(self.position - position) < 1.5:
                    check_threat.append(position)

        # if pawn
        elif self.piece_type == PAWN_NAMES[0].split(sep="_")[0]:
            try:
                self.board.cells[tuple(self.position + self.possible_step_directions['capture_left'])]
                check_threat.append(self.position + self.possible_step_directions['capture_left'])
            except KeyError:
                pass

            try:
                self.board.cells[tuple(self.position + self.possible_step_directions['capture_right'])]
                check_threat.append(self.position + self.possible_step_directions['capture_right'])
            except KeyError:
                pass

        # if knight
        elif self.piece_type == KNIGHT_NAMES[0].split(sep="_")[0]:
            for dir, possible_step in self.possible_step_directions.items():
                try:
                    self.board.cells[tuple(self.position + possible_step)]
                    check_threat.append(self.position + possible_step)
                except KeyError:
                    pass

        # if not king, pawn or knight
        else:
            avail, ch = self.get_available_positions()
            check_threat = avail + ch
        return check_threat

    def giving_check(self):
        opponent = self.board.player_1 if self.color != self.board.player_1.color else self.board.player_2
        check_threat = self.positions_kept_in_check()
        tuplized_check_threat = [tuple(i) for i in check_threat]
        if tuple(opponent.king.position) in tuplized_check_threat:
            return True
        else:
            return False

    def remove_piece_ability(self, piece):
        avail, _ = self.get_available_positions()
        if tuple(piece.position) in [tuple(i) for i in avail]:
            # if we are blocking a check
            if self.blocking_check(from_piece=piece):
                return True
            # if we are not blocking a check
            else:
                return True

    def get_opponent_pieces_blocking_opponent_king(self):
        return []

    def smaller_empty(self, smaller_step_positions):
        smaller_empty = [0 if self.board.cells[tuple(smaller_pos)] is None else 1 for smaller_pos in
                         smaller_step_positions]
        if sum(smaller_empty) == 0:
            return True

    def get_available_positions(self):
        """returns all available new positions (not step directions)"""

        available_new_positions = []
        check_threat = []
        opponent = self.board.player_1 if self.color != self.board.player_1.color else self.board.player_2

        # direction is forward_right, backward_right ...
        for direction in self.possible_step_directions.keys():

            for step in self.possible_step_directions[direction]:
                # new_position: the position we are examining
                new_position = self.position + step
                # smaller_step_positions: the positions which are closer to the current position, later check if these are empty
                smaller_step_positions = []
                for s in self.possible_step_directions[direction]:
                    if np.linalg.norm(s) < np.linalg.norm(step):
                        smaller_step_positions.append(self.position + s)

                # on the board
                try:
                    # empty
                    if self.board.cells[tuple(new_position)] is None:
                        # all smaller positions are empty
                        if self.smaller_empty(smaller_step_positions):
                            available_new_positions.append(new_position)

                    # not empty
                    else:
                        # other player's piece
                        if self.board.cells[tuple(new_position)].color != self.color:
                            # all smaller positions are empty
                            if self.smaller_empty(smaller_step_positions):
                                available_new_positions.append(new_position)

                        # our piece -- append to positions_kept_in_check
                        else:
                            # all smaller positions are empty
                            if self.smaller_empty(smaller_step_positions):
                                check_threat.append(new_position)

                except KeyError:
                    # out-of-bound step
                    pass

        return available_new_positions, check_threat




    def get_opponent_pieces_blocking_opponent_king(self):
        """returns opponent.pieces if those are standing between self and opponent.king"""
        intermediate_opponent_pieces = []
        opponent = self.board.player_1 if self.color != self.board.player_1.color else self.board.player_2

        # direction is forward_right, backward_right ...
        for direction in self.possible_step_directions.keys():
            available_positions_per_direction = []
            for step in self.possible_step_directions[direction]:
                # new_position: the position we are examining
                new_position = self.position + step

                # on the board
                try:
                    # empty
                    if self.board.cells[tuple(new_position)] is None:
                        available_positions_per_direction.append(new_position)

                    # not empty
                    else:
                        # other player's piece
                        if self.board.cells[tuple(new_position)].color != self.color:
                            available_positions_per_direction.append(new_position)

                except KeyError:
                    # out-of-bound step
                    pass

                tuplized_positions = [tuple(i) for i in available_positions_per_direction]

                if tuple(opponent.king.position) in tuplized_positions:
                    for opp_piece in opponent.pieces:
                        if tuple(opp_piece.position) in tuplized_positions:
                            if opp_piece.piece_type != opponent.king.piece_type:
                                intermediate_opponent_pieces.append(opp_piece)
                    return intermediate_opponent_pieces
        return intermediate_opponent_pieces


class Pawn(Piece):
    def __init__(self, name, color):
        self.name = name
        self.piece_type = self.name.split(sep="_")[0]
        self.color = color
        self.starting_position = STARTING_POSITION[(name, color)]
        self.position = STARTING_POSITION[(name, color)]
        self.board = None
        self.white_steps = {
            'normal_step': np.array([0, 1]),
            'capture_right': np.array([1, 1]),
            'capture_left': np.array([-1, 1]),
            'initial_long_step': np.array([0, 2])
        }
        self.black_steps = {
            'normal_step': np.array([0, -1]),
            'capture_right': np.array([-1, -1]),
            'capture_left': np.array([1, -1]),
            'initial_long_step': np.array([0, -2])
        }
        self.possible_step_directions = self.white_steps if self.color == "white" else self.black_steps

    def promote(self, new_position, verbose):
        active_player = self.board.player_1 if self.color == self.board.player_1.color else self.board.player_2
        current_position = self.position
        if self.color == "white":
            if tuple(new_position) in [(1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8)]:

                active_player_picece_types = [p.piece_type for p in active_player.pieces]
                if active_player_picece_types.count("queen") == 0:
                    active_player.pop_piece(self, verbose)
                    active_player.pieces.append(Queen(name="queen_new", color=active_player.color))
                elif active_player_picece_types.count("rook") == 1:
                    active_player.pop_piece(self, verbose)
                    active_player.pieces.append(Rook(name="rook_new", color=active_player.color))
                elif active_player_picece_types.count("bishop") == 1:
                    active_player.pop_piece(self, verbose)
                    active_player.pieces.append(Bishop(name="bishop_new", color=active_player.color))
                elif active_player_picece_types.count("knight") == 1:
                    active_player.pop_piece(self, verbose)
                    active_player.pieces.append(Knight(name="knight_new", color=active_player.color))
                else:
                    # do not remove current pawn
                    pass

                for p in active_player.pieces:
                    if p.name.endswith("new"):
                        if tuple(p.position) == (0, 0):
                            p.position = new_position
                            p.board = active_player.board

    def get_available_positions(self):
        """returns all available new positions (not step directions)"""
        available_new_positions = []
        opponent = self.board.player_1 if self.color != self.board.player_1.color else self.board.player_2
        # We can move forward ONE cell if that cell is empty and we are not blocking a check currently
        try:
            if self.board.cells[tuple(self.position + self.possible_step_directions['normal_step'])] is None:
                for opp_piece in opponent.pieces:
                    if not self.blocking_check(from_piece=opp_piece):
                        available_new_positions.append(self.position + self.possible_step_directions['normal_step'])
        except KeyError:
            pass

        # We can capture RIGHT if the opponents piece is there and we are not blocking check currently
        try:
            if self.board.cells[tuple(self.position + self.possible_step_directions['capture_right'])] is not None:
                if self.board.cells[
                    tuple(self.position + self.possible_step_directions['capture_right'])].color != self.color:
                    for opp_piece in opponent.pieces:
                        if not self.blocking_check(from_piece=opp_piece):
                            available_new_positions.append(self.position + self.possible_step_directions['capture_right'])
        except KeyError:
            pass

        # We can capture LEFT if the opponents piece is there and we are not blocking check currently
        try:
            if self.board.cells[tuple(self.position + self.possible_step_directions['capture_left'])] is not None:
                if self.board.cells[
                    tuple(self.position + self.possible_step_directions['capture_left'])].color != self.color:
                    for opp_piece in opponent.pieces:
                        if not self.blocking_check(from_piece=opp_piece):
                            available_new_positions.append(self.position + self.possible_step_directions['capture_left'])
        except KeyError:
            pass

        # We can move forward TWO cells if that cell is empty, we haven't moved yet and we are not blocking a check currently
        try:
            if self.board.cells[tuple(self.position + self.possible_step_directions['initial_long_step'])] is None:
                if self.board.cells[tuple(self.position + self.possible_step_directions['normal_step'])] is None:
                    if np.all(self.position == self.starting_position):
                        for opp_piece in opponent.pieces:
                            if not self.blocking_check(from_piece=opp_piece):
                                available_new_positions.append(
                                    self.position + self.possible_step_directions['initial_long_step'])
        except KeyError:
            pass
        return available_new_positions, []


class King(Piece):
    def __init__(self, name, color):
        self.name = name
        self.piece_type = self.name.split(sep="_")[0]
        self.color = color
        self.starting_position = STARTING_POSITION[(name, color)]
        self.position = STARTING_POSITION[(name, color)]
        self.board = None
        self.white_steps = {
            'forward': np.array([0, 1]),
            'forward_right': np.array([1, 1]),
            'forward_left': np.array([-1, 1]),
            'right': np.array([1, 0]),
            'left': np.array([-1, 0]),
            'backward': np.array([0, -1]),
            'backward_right': np.array([1, -1]),
            'backward_left': np.array([-1, -1])
        }
        self.black_steps = {
            'forward': np.array([0, -1]),
            'forward_right': np.array([-1, -1]),
            'forward_left': np.array([1, -1]),
            'right': np.array([-1, 0]),
            'left': np.array([1, 0]),
            'backward': np.array([0, 1]),
            'backward_right': np.array([-1, 1]),
            'backward_left': np.array([1, 1])
        }
        self.possible_step_directions = self.white_steps if self.color == "white" else self.black_steps

    def get_available_positions(self):
        """returns all available new positions (not step directions)"""

        # castling not implemented yet !!

        available_new_positions = []
        step_directions = list(self.possible_step_directions.values())

        positions_in_check = self.get_positions_in_check()
        positions_in_check = [tuple(item) for item in positions_in_check]

        # We can move one cell if that cell is empty or opponent's (but not the opponent king's) and it is not in check_threat
        for direction in step_directions:
            new_position = self.position + direction

            # on the board
            try:
                # empty
                if self.board.cells[tuple(new_position)] is None:
                    # not in check
                    if tuple(new_position) not in positions_in_check:
                        # still far enough from other king
                        if self.other_king_distance(
                                new_position) > 1.5:  # a little bigger than math.sqrt(2) to circumvent math and numpy difference
                            available_new_positions.append(new_position)
                # other player's piece
                else:
                    if self.board.cells[tuple(new_position)].color != self.color:
                        # not in check
                        if tuple(new_position) not in positions_in_check:
                            # still far enough from other king
                            if self.other_king_distance(new_position) > 1.5:
                                available_new_positions.append(new_position)


            except KeyError:
                # out-of-bound step
                pass
        if np.any([np.any(item == 9) for item in available_new_positions]):
            print("out of bounds")
        return available_new_positions, []

    def king_in_check(self):
        positions_in_check = self.get_positions_in_check()
        positions_in_check = [tuple(item) for item in positions_in_check]
        if tuple(self.position) in positions_in_check:
            return True
        else:
            return False

    def other_king_distance(self, from_position):
        opponent = self.board.player_1 if self.color != self.board.player_1.color else self.board.player_2
        opponent_king = None

        for p in opponent.pieces:
            if p.piece_type == self.piece_type:
                opponent_king = p

        # if opponent king remains None I will get an error
        distance = np.linalg.norm(opponent_king.position - from_position)
        return distance


class Bishop(Piece):
    def __init__(self, name, color):
        self.name = name
        self.piece_type = self.name.split(sep="_")[0]
        self.color = color
        self.starting_position = STARTING_POSITION[(name, color)]
        self.position = STARTING_POSITION[(name, color)]
        self.board = None
        self.white_steps = {
            'forward_right': [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4]),
                              np.array([5, 5]), np.array([6, 6]), np.array([7, 7])],

            'forward_left': [np.array([-1, 1]), np.array([-2, 2]), np.array([-3, 3]), np.array([-4, 4]),
                             np.array([-5, 5]), np.array([-6, 6]), np.array([-7, 7])],

            'backward_right': [np.array([1, -1]), np.array([2, -2]), np.array([3, -3]), np.array([4, -4]),
                               np.array([5, -5]), np.array([6, -6]), np.array([7, -7])],

            'backward_left': [np.array([-1, 1]), np.array([-2, 2]), np.array([-3, 3]), np.array([-4, 4]),
                              np.array([-5, 5]), np.array([-6, 6]), np.array([-7, 7])]
        }
        self.black_steps = {
            'forward_right': [np.array([-1, 1]), np.array([-2, 2]), np.array([-3, 3]), np.array([-4, 4]),
                              np.array([-5, 5]), np.array([-6, 6]), np.array([-7, 7])],

            'forward_left': [np.array([1, -1]), np.array([2, -2]), np.array([3, -3]), np.array([4, -4]),
                             np.array([5, -5]), np.array([6, -6]), np.array([7, -7])],

            'backward_right': [np.array([-1, 1]), np.array([-2, 2]), np.array([-3, 3]), np.array([-4, 4]),
                               np.array([-5, 5]), np.array([-6, 6]), np.array([-7, 7])],

            'backward_left': [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4]),
                              np.array([5, 5]), np.array([6, 6]), np.array([7, 7])],
        }

        self.possible_step_directions = self.white_steps if self.color == "white" else self.black_steps


class Rook(Piece):
    def __init__(self, name, color):
        self.name = name
        self.piece_type = self.name.split(sep="_")[0]
        self.color = color
        self.starting_position = STARTING_POSITION[(name, color)]
        self.position = STARTING_POSITION[(name, color)]
        self.board = None
        self.white_steps = {
            'forward': [np.array([0, 1]), np.array([0, 2]), np.array([0, 3]), np.array([0, 4]),
                        np.array([0, 5]), np.array([0, 6]), np.array([0, 7])],
            'backward': [np.array([0, -1]), np.array([0, -2]), np.array([0, -3]), np.array([0, -4]),
                         np.array([0, -5]), np.array([0, -6]), np.array([0, -7])],
            'right': [np.array([1, 0]), np.array([2, 0]), np.array([3, 0]), np.array([4, 0]),
                      np.array([5, 0]), np.array([6, 0]), np.array([7, 0])],
            'left': [np.array([-1, 0]), np.array([-2, 0]), np.array([-3, 0]), np.array([-4, 0]),
                     np.array([-5, 0]), np.array([-6, 0]), np.array([-7, 0])]
        }
        self.black_steps = {
            'forward': [np.array([0, -1]), np.array([0, -2]), np.array([0, -3]), np.array([0, -4]),
                        np.array([0, -5]), np.array([0, -6]), np.array([0, -7])],
            'backward': [np.array([0, 1]), np.array([0, 2]), np.array([0, 3]), np.array([0, 4]),
                         np.array([0, 5]), np.array([0, 6]), np.array([0, 7])],
            'right': [np.array([-1, 0]), np.array([-2, 0]), np.array([-3, 0]), np.array([-4, 0]),
                      np.array([-5, 0]), np.array([-6, 0]), np.array([-7, 0])],
            'left': [np.array([1, 0]), np.array([2, 0]), np.array([3, 0]), np.array([4, 0]),
                     np.array([5, 0]), np.array([6, 0]), np.array([7, 0])]
        }

        self.possible_step_directions = self.white_steps if self.color == "white" else self.black_steps


class Queen(Piece):
    def __init__(self, name, color):
        self.name = name
        self.piece_type = self.name.split(sep="_")[0]
        self.color = color
        self.starting_position = STARTING_POSITION[(name, color)]
        self.position = STARTING_POSITION[(name, color)]
        self.board = None
        self.white_steps = {
            'forward': [np.array([0, 1]), np.array([0, 2]), np.array([0, 3]), np.array([0, 4]),
                        np.array([0, 5]), np.array([0, 6]), np.array([0, 7])],
            'backward': [np.array([0, -1]), np.array([0, -2]), np.array([0, -3]), np.array([0, -4]),
                         np.array([0, -5]), np.array([0, -6]), np.array([0, -7])],
            'right': [np.array([1, 0]), np.array([2, 0]), np.array([3, 0]), np.array([4, 0]),
                      np.array([5, 0]), np.array([6, 0]), np.array([7, 0])],
            'left': [np.array([-1, 0]), np.array([-2, 0]), np.array([-3, 0]), np.array([-4, 0]),
                     np.array([-5, 0]), np.array([-6, 0]), np.array([-7, 0])],
            'forward_right': [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4]),
                              np.array([5, 5]), np.array([6, 6]), np.array([7, 7])],
            'forward_left': [np.array([-1, 1]), np.array([-2, 2]), np.array([-3, 3]), np.array([-4, 4]),
                             np.array([-5, 5]), np.array([-6, 6]), np.array([-7, 7])],
            'backward_right': [np.array([1, -1]), np.array([2, -2]), np.array([3, -3]), np.array([4, -4]),
                               np.array([5, -5]), np.array([6, -6]), np.array([7, -7])],
            'backward_left': [np.array([-1, 1]), np.array([-2, 2]), np.array([-3, 3]), np.array([-4, 4]),
                              np.array([-5, 5]), np.array([-6, 6]), np.array([-7, 7])]
        }
        self.black_steps = {
            'forward': [np.array([0, -1]), np.array([0, -2]), np.array([0, -3]), np.array([0, -4]),
                        np.array([0, -5]), np.array([0, -6]), np.array([0, -7])],
            'backward': [np.array([0, 1]), np.array([0, 2]), np.array([0, 3]), np.array([0, 4]),
                         np.array([0, 5]), np.array([0, 6]), np.array([0, 7])],
            'right': [np.array([-1, 0]), np.array([-2, 0]), np.array([-3, 0]), np.array([-4, 0]),
                      np.array([-5, 0]), np.array([-6, 0]), np.array([-7, 0])],
            'left': [np.array([1, 0]), np.array([2, 0]), np.array([3, 0]), np.array([4, 0]),
                     np.array([5, 0]), np.array([6, 0]), np.array([7, 0])],
            'forward_right': [np.array([-1, 1]), np.array([-2, 2]), np.array([-3, 3]), np.array([-4, 4]),
                              np.array([-5, 5]), np.array([-6, 6]), np.array([-7, 7])],
            'forward_left': [np.array([1, -1]), np.array([2, -2]), np.array([3, -3]), np.array([4, -4]),
                             np.array([5, -5]), np.array([6, -6]), np.array([7, -7])],
            'backward_right': [np.array([-1, 1]), np.array([-2, 2]), np.array([-3, 3]), np.array([-4, 4]),
                               np.array([-5, 5]), np.array([-6, 6]), np.array([-7, 7])],
            'backward_left': [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4]),
                              np.array([5, 5]), np.array([6, 6]), np.array([7, 7])]
        }

        self.possible_step_directions = self.white_steps if self.color == "white" else self.black_steps

class Knight(Piece):
    def __init__(self, name, color):
        self.name = name
        self.piece_type = self.name.split(sep="_")[0]
        self.color = color
        self.starting_position = STARTING_POSITION[(name, color)]
        self.position = STARTING_POSITION[(name, color)]
        self.board = None
        self.white_steps = {
            'forward_right': np.array([1, 2]),
            'backward_right': np.array([1, -2]),
            'forward_left': np.array([-1, 2]),
            'backward_left': np.array([-1, -2]),
            'right_forward': np.array([2, 1]),
            'left_forward': np.array([-2, 1]),
            'right_backward': np.array([2, -1]),
            'left_backward': np.array([-2, -1])
        }
        self.black_steps = {
            'forward_right': np.array([-1, -2]),
            'backward_right': np.array([-1, 2]),
            'forward_left': np.array([1, -2]),
            'backward_left': np.array([1, 2]),
            'right_forward': np.array([-2, -1]),
            'left_forward': np.array([2, -1]),
            'right_backward': np.array([-2, 1]),
            'left_backward': np.array([2, 1])
        }

        self.possible_step_directions = self.white_steps if self.color == "white" else self.black_steps

    def get_available_positions(self):
        """returns all available new positions (not step directions)"""

        available_new_positions = []
        check_threat = []

        # direction is forward_right, backward_right ...
        for direction in self.possible_step_directions.keys():
            # new_position: the position we are examining
            new_position = self.position + self.possible_step_directions[direction]
            # on the board
            try:
                # empty
                if self.board.cells[tuple(new_position)] is None:
                    available_new_positions.append(new_position)

                # not empty
                else:
                    # other player's piece
                    if self.board.cells[tuple(new_position)].color != self.color:
                        available_new_positions.append(new_position)

                    # our piece -- append to positions_kept_in_check
                    else:
                        check_threat.append(new_position)

            except KeyError:
                # out-of-bound step
                pass

        return available_new_positions, check_threat


class Player:
    def __init__(self, color):
        self.color = color
        # self.pieces = [Piece(name=name, color=self.color) for name in PIECE_NAMES]
        self.pawns = [Pawn(name=pawn_name, color=self.color) for pawn_name in PAWN_NAMES]
        self.king = King(name=KING_NAME, color=self.color)
        self.queen = Queen(name=QUEEN_NAME, color=self.color)
        self.bishops = [Bishop(name=bishop_name, color=self.color) for bishop_name in BISHOP_NAMES]
        self.rooks = [Rook(name=rook_name, color=self.color) for rook_name in ROOK_NAMES]
        self.knights = [Knight(name=knight_name, color=self.color) for knight_name in KNIGHT_NAMES]
        self.pieces = self.pawns + self.bishops + self.rooks + self.knights + [self.king] + [self.queen]
        self.active = False
        self.board = None

    def __repr__(self):
        return self.color

    def get_piece(self, position):
        for piece in self.pieces:
            if np.all(piece.position == position):
                return piece
        return None

    def pop_piece(self, piece, verbose):
        if piece.piece_type != 'king':
            popped = self.pieces.pop(self.pieces.index(piece))
            if verbose:
                print(popped)
        else:
            #if piece.color != self.color:
            print(self.board)
            raise ValueError("Tried to remove king")

    def get_available_pieces(self):
        available = []
        for piece in self.pieces:
            avail, _ = piece.get_available_positions()
            if avail:
                available.append(piece)
        return available

    def choose_piece(self):
        if self.king.king_in_check():
            print("king in check")
            return self.king
        else:
            available_pieces = self.get_available_pieces()
            if available_pieces:
                return random.choice(available_pieces)
            else:
                return []

    def piece_giving_check(self):
        for p in self.pieces:
            if p.giving_check():
                return p

    @property
    def losing(self):
        opponent = self.board.player_1 if self.color != self.board.player_1.color else self.board.player_2
        threatening_pieces = []
        if self.king.king_in_check():
            check_giving_piece = opponent.piece_giving_check()

            for p in self.pieces:
                if not p.remove_piece_ability(check_giving_piece):
                    avail, _ = self.king.get_available_positions()
                    if avail:
                        return False
                    else:
                        print(f"{self} is losing")
                        print(self.board)
                        return True


class Board:
    def __init__(self, player_1, player_2):
        self.all_positions = list(itertools.product(range(1, 9), range(1, 9)))
        self.player_1 = player_1
        self.player_2 = player_2
        self.players = [player_1, player_2]
        # set board for all players and pieces
        for player in self.players:
            player.board = self
            for piece in player.pieces:
                piece.board = self

        self.cells = {pos: None for pos in self.all_positions}
        self.update_board()

    def __repr__(self):
        vertical = "|"
        horizontal = "-----"
        empty = "    "
        new_line = "\n"
        result = ""

        cell_number_y = 1
        for y in range(0, 17):
            if y % 2 == 0:
                line = horizontal * 8 + new_line
                result = line + result
            else:
                line_segment = ""
                cell_number_x = 1
                for x in range(0, 18):
                    if x % 2 == 0:
                        line_segment = line_segment + vertical
                    else:
                        try:
                            if self.cells[(cell_number_x, cell_number_y)] is None:
                                line_segment = line_segment + empty
                                cell_number_x += 1
                            else:
                                line_segment = line_segment + self.cells[(cell_number_x, cell_number_y)].__repr__()
                                cell_number_x += 1
                        except KeyError:
                            # we have to stop at cell_number_x == 8 so that we do not get a key error
                            pass
                cell_number_y += 1

                line = line_segment + new_line
                result = line + result
        return result

    def update_board(self):
        self.cells = {pos: None for pos in self.all_positions}
        for player in self.players:
            for piece in player.pieces:
                self.cells[tuple(piece.position)] = piece

    def play(self, show=True, verbose=True):
        # limiter is only for testing until player.losing is not functioning
        limiter = 0

        i = 0
        run = True
        while run:
            if i % 2 == 0:
                active_player = self.player_1
                opponent = self.player_2
            else:
                active_player = self.player_2
                opponent = self.player_1

            active_piece = active_player.choose_piece()

            if active_piece:
                new_position = active_piece.choose_new_position()
                # if opponent is there -- remove that piece from the board
                capture = opponent.get_piece(new_position)
                if capture is not None:
                    opponent.pop_piece(capture, verbose)
                active_piece.move(to=new_position)
                active_piece.promote(new_position, verbose)

            else:
                print("Finished")
                break

            self.update_board()
            if show:
                print(self)
                time.sleep(2)

            # only for testing
            limiter += 1
            if limiter == 1000:
                print(limiter)
                print(self)
                break

            # increment i
            i += 1

            run = not np.any([self.player_1.losing, self.player_2.losing])
            if not run:
                print(i)


def main():
    # initiate board
    player_1 = Player("white")
    player_2 = Player("black")
    board = Board(player_1, player_2)
    board.play(show=False)


if __name__ == '__main__':
    main()
