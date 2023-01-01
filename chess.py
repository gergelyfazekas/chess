import math
import random

import numpy as np
import itertools

PAWN_NAMES = ['pawn_1', 'pawn_2', 'pawn_3', 'pawn_4', 'pawn_5', 'pawn_6', 'pawn_7', 'pawn_8']
KING_NAME = 'king'

# starting position using x-y coordinates with x, y in [1,8]
STARTING_POSITION = {
    ('king', 'white'): np.array([5, 1]),
    ('pawn_1', 'white'): np.array([1, 2]),
    ('pawn_2', 'white'): np.array([2, 2]),
    ('pawn_3', 'white'): np.array([3, 2]),
    ('pawn_4', 'white'): np.array([4, 2]),
    ('pawn_5', 'white'): np.array([5, 2]),
    ('pawn_6', 'white'): np.array([6, 2]),
    ('pawn_7', 'white'): np.array([7, 2]),
    ('pawn_8', 'white'): np.array([8, 2]),

    ('king', 'black'): np.array([5, 8]),
    ('pawn_1', 'black'): np.array([1, 7]),
    ('pawn_2', 'black'): np.array([2, 7]),
    ('pawn_3', 'black'): np.array([3, 7]),
    ('pawn_4', 'black'): np.array([4, 7]),
    ('pawn_5', 'black'): np.array([5, 7]),
    ('pawn_6', 'black'): np.array([6, 7]),
    ('pawn_7', 'black'): np.array([7, 7]),
    ('pawn_8', 'black'): np.array([8, 7])
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

    def choose_new_position(self):
        """handles all new-position-selecting procedures the returned value can be fed to self.move()"""
        available_positions = self.get_available_positions()
        if available_positions:
            return random.choice(available_positions)
        else:
            return []

    @property
    def blocking_check(self):
        # True if the piece's current position blocks a check, else False
        # If blocking_check is True ––> we cannot step out of that position
        pass

    def drop_out_of_bounds_positions(self, positions):
        valid_positions = []
        for pos in positions:
            if pos[0] in range(1, 9):
                if pos[1] in range(1, 9):
                    valid_positions.append(pos)

        return valid_positions


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

    def get_available_positions(self):
        """returns all available new positions (not step directions)"""
        available_new_positions = []

        # We can move forward ONE cell if that cell is empty and we are not blocking a check currently
        try:
            if self.board.cells[tuple(self.position + self.possible_step_directions['normal_step'])] is None:
                if not self.blocking_check:
                    available_new_positions.append(self.position + self.possible_step_directions['normal_step'])
        except KeyError:
            pass

        # We can capture RIGHT if the opponents piece is there and we are not blocking check currently
        try:
            if self.board.cells[tuple(self.position + self.possible_step_directions['capture_right'])] is not None:
                if self.board.cells[
                    tuple(self.position + self.possible_step_directions['capture_right'])].color != self.color:
                    if not self.blocking_check:
                        available_new_positions.append(self.position + self.possible_step_directions['capture_right'])
        except KeyError:
            pass

        # We can capture LEFT if the opponents piece is there and we are not blocking check currently
        try:
            if self.board.cells[tuple(self.position + self.possible_step_directions['capture_left'])] is not None:
                if self.board.cells[
                    tuple(self.position + self.possible_step_directions['capture_left'])].color != self.color:
                    if not self.blocking_check:
                        available_new_positions.append(self.position + self.possible_step_directions['capture_left'])
        except KeyError:
            pass

        # We can move forward TWO cells if that cell is empty, we haven't moved yet and we are not blocking a check currently
        try:
            if self.board.cells[tuple(self.position + self.possible_step_directions['initial_long_step'])] is None:
                if np.all(self.position == self.starting_position):
                    if not self.blocking_check:
                        available_new_positions.append(
                            self.position + self.possible_step_directions['initial_long_step'])
        except KeyError:
            pass

        return available_new_positions

    def positions_kept_in_check(self):
        check_threat = []
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
        return check_threat


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
                        if self.other_king_distance(new_position) > 1.5:  # a little bigger than math.sqrt(2) to circumvent math and numpy difference
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
        if np.any([np.any(item==9) for item in available_new_positions]):
            print("out of bounds")
        return available_new_positions

    def get_positions_in_check(self):
        opponent = self.board.player_1 if self.color != self.board.player_1.color else self.board.player_2
        positions_in_check = []
        for p in opponent.pieces:
            positions_in_check.extend(p.positions_kept_in_check())
        return positions_in_check

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

    def positions_kept_in_check(self):
        check_threat = []
        for position in self.board.all_positions:
            if np.linalg.norm(self.position - position) < 1.5:
                check_threat.append(position)

        return check_threat


class Player:
    def __init__(self, color):
        self.color = color
        # self.pieces = [Piece(name=name, color=self.color) for name in PIECE_NAMES]
        self.pawns = [Pawn(name=pawn_name, color=self.color) for pawn_name in PAWN_NAMES]
        self.king = King(name=KING_NAME, color=self.color)
        self.pieces = self.pawns + [self.king]
        self.active = False
        self.board = None

    def __repr__(self):
        return self.color

    def get_piece(self, position):
        for piece in self.pieces:
            if np.all(piece.position == position):
                return piece
        return None

    def pop_piece(self, piece):
        if piece.piece_type != 'king':
            self.pieces.pop(self.pieces.index(piece))
        else:
            raise ValueError("Tried to remove king")

    def get_available_pieces(self):
        available = []
        for piece in self.pieces:
            if piece.get_available_positions():
                available.append(piece)
        return available

    def choose_piece(self):
        if self.king.king_in_check():
            return self.king
        else:
            available_pieces = self.get_available_pieces()
            if available_pieces:
                return random.choice(available_pieces)
            else:
                return []

    @property
    def losing(self):
        if self.king.get_available_positions():
            return False
        else:
            print(f"{self} is losing")
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

    def play(self, show=True):
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
                    opponent.pop_piece(capture)
                active_piece.move(to=new_position)
            else:
                # print("active_player", active_player)
                # print("active_piece", active_piece)
                # print("available_positions", active_piece.get_available_positions())
                print("Finished")
                break

            self.update_board()
            if show:
                print(self)

            # only for testing
            limiter += 1
            if limiter == 700:
                break

            # increment i
            i += 1

            run = not np.any([self.player_1.losing, self.player_2.losing])


def main():
    # initiate board
    player_1 = Player("white")
    player_2 = Player("black")
    board = Board(player_1, player_2)
    board.play()


if __name__ == '__main__':
    main()
