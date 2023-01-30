import pickle
import random
import time
import numpy as np
import itertools
from deep_learning import NeuralNet

# Setting seed for reproducibility
SEED = 2
random.seed(SEED)

# Piece names
KING_NAME = 'king'
QUEEN_NAME = 'queen'
KNIGHT_NAMES = ['knight_1', 'knight_2']
ROOK_NAMES = ['rook_1', 'rook_2']
BISHOP_NAMES = ['bishop_1', 'bishop_2']
PAWN_NAMES = ['pawn_1', 'pawn_2', 'pawn_3', 'pawn_4', 'pawn_5', 'pawn_6', 'pawn_7', 'pawn_8']

# Starting position using x-y coordinates with x, y in [1,8]
# White and black are mirrored, e.g. bishop_1 is on x=3 for both teams (they face each other at start)
with open("piece_starting_positions.pickle", "rb") as f:
    STARTING_POSITIONS = pickle.load(f)

# Standard piece values from wikipedia -- Chess bot only
with open("piece_values.pickle", "rb") as f:
    PIECE_VALUES = pickle.load(f)

# Fixed id for each piece (as well as for None)
# This is an easy representation of a cell's content
with open("piece_ids.pickle", "rb") as f:
    PIECE_IDS = pickle.load(f)


class Piece:
    def __init__(self, name, color):
        self.name = name
        self.player = None
        self.opponent = None
        self.piece_type = self.name.split(sep="_")[0]
        self.color = color
        self.starting_position = STARTING_POSITIONS[(name, color)]
        self.position = STARTING_POSITIONS[(name, color)]
        self.value = PIECE_VALUES[(name, color)]
        self.board = None
        self.possible_step_directions = None
        self.id = min([item[0] if item[1].__str__() == self.__str__() else 1000 for item in PIECE_IDS])
        # set_step_directions is implemented in the Pawn, King, Queen, ... classes
        self.set_step_directions()

    def __repr__(self):
        """first letter of self.color + first letter of self.name + last letter of self.name
        Examples:
        w_p1: white pawn_1
        w_kg: white king
        b_k2: black knight_2
        b_qw: black queen new (pawn was promoted for a new queen)
        """
        return f"{self.color[0]}_{self.name[0]}{self.name[-1]}"

    def __str__(self):
        """first letter of self.color + first letter of self.name + last letter of self.name
        Examples:
        w_p1: white pawn_1
        w_kg: white king
        b_k2: black knight_2
        b_qw: black queen new (pawn was promoted for a new queen)
        """
        return f"{self.color[0]}_{self.name[0]}{self.name[-1]}"

    @staticmethod
    def remove_duplicate_positions(list_of_positions):
        """removes duplicate positions by turning them into a set and back to a list of np.arrays"""
        if list_of_positions:
            l = [tuple(item) for item in list_of_positions]
            l = set(l)
            l = [np.array(item) for item in l]
        else:
            l = []
        return l

    def set_step_directions(self):
        """Implemented at each piece subclass: see for example Pawn.set_step_directions"""
        pass

    def move(self, to, remember=True):
        """moves a piece to a new position (and also )
        args:
        to: tuple, the new position as a tuple of (x,y) coordinates
        remember: bool, if True appends the move to the move history of self.board (has to be True for the chess bot)

        returns:
        self, old_pos, new_pos -- for tracking the steps made during a game
        """
        old_pos = self.position
        self.position = to
        move = (self, old_pos, self.position)
        if remember:
            self.board.move_history.append(move)
        return move

    def promote(self, new_position, verbose=False):
        """Implemented only for Pawn"""
        pass

    def blocking_check(self, from_piece):
        """returns if we are blocking check (bool) from a particular piece
        args:
        from_piece: Piece instance, would this piece cause a check if we were not at this position

        returns:
        bool, True if we are blocking a check from 'from_piece' (if we are king we are not blocking check)
        """
        positions_in_direction_of_king = from_piece.get_closer_positions_in_direction(self.player.king.position)
        if self.piece_type == KING_NAME:
            return False
        else:
            if from_piece.piece_type in ("knight", "pawn"):
                # cannot block a check for a knight or a pawn by their nature
                return False
            else:
                if positions_in_direction_of_king:
                    # all closer are empty except for us and our king
                    if sum([0 if self.board.cells[tuple(pos)] is None else 1 for pos in
                            positions_in_direction_of_king]) == 2:

                        if tuple(self.position) in [tuple(item) for item in positions_in_direction_of_king]:
                            return True
                        else:
                            return False
                    else:
                        return False
                else:
                    return False

    def get_positions_in_check(self):
        """get all positions unavailable for the king to step into as one opponent piece keeps it in check"""
        positions_in_check = []
        
        for p in self.opponent.pieces:
            in_check = p.positions_kept_in_check()
            
            if in_check:
                positions_in_check.extend(in_check)

        positions_in_check = self.remove_duplicate_positions(positions_in_check)
        return positions_in_check

    def positions_kept_in_check(self):
        """returns all positions that we (one piece) are keeping in check (not all pieces like get_positions_in_check)"""
        check_threat = []
        
        # if king
        if self.piece_type == KING_NAME:
            possible_positions = self.convert_steps_to_positions()
            for new_position in possible_positions:
                check_threat.append(new_position)

        # if pawn
        elif self.piece_type == PAWN_NAMES[0].split(sep="_")[0]:
            try:
                self.board.cells[tuple(self.position + self.possible_step_directions['capture_left'][0])]
                check_threat.append(self.position + self.possible_step_directions['capture_left'][0])
            except KeyError:
                pass

            try:
                self.board.cells[tuple(self.position + self.possible_step_directions['capture_right'][0])]
                check_threat.append(self.position + self.possible_step_directions['capture_right'][0])
            except KeyError:
                pass

        # if knight
        elif self.piece_type == KNIGHT_NAMES[0].split(sep="_")[0]:
            check_threat = self.convert_steps_to_positions()

        # if not king, pawn or knight
        else:
            possible_positions = self.convert_steps_to_positions()

            for new_position in possible_positions:
                # if all closer positions to us are also empty
                closer_positions = self.get_closer_positions_in_direction(new_position)
                # closer positions is a list but all cells are empty
                if closer_positions:
                    if sum([0 if self.board.cells[tuple(pos)] is None else 1 for pos in
                            closer_positions]) == 0:
                        check_threat.append(new_position)

                    # strictly closer is empty
                    elif sum([0 if self.board.cells[tuple(pos)] is None else 1 for pos in
                              closer_positions][:-1]) == 0:

                        # last is not empty
                        if [0 if self.board.cells[tuple(pos)] is None else 1 for pos in
                            closer_positions][-1] == 1:
                            check_threat.append(new_position)

        check_threat = self.remove_duplicate_positions(check_threat)
        return check_threat

    def giving_check(self):
        """returns True if the current piece causes check for the opponent's king (else False)"""
        check_threat = self.positions_kept_in_check()
        check_threat = [tuple(i) for i in check_threat]
        if tuple(self.opponent.king.position) in check_threat:
            return True
        else:
            return False

    def capture_ability(self, piece):
        """returns True if the current piece can capture a particular piece of the opponent (else False)
        args:
        piece: Piece instance, the opponent's piece we are interested in capturing
        """
        legal_moves = self.get_legal_positions()
        legal_moves = [tuple(item) for item in legal_moves]
        if tuple(piece.position) in legal_moves:
            return True
        else:
            return False

    def _get_steps_in_direction(self, step):
        """returns the steps in the direction of a given step, sorted in ascending order (by norm)
        args:
        step: np.array, e.g. a normal step for a white pawn is np.array([0,1])
        
        example usage:
        result = Piece("pawn_1", "white")._get_steps_in_direction(step=np.array([0,1])
        print(result)
            out: [np.array([0,1]), np.array([0,2])]
        """
        steps_in_direction = []
        possible_directions = list(self.possible_step_directions.keys())
        # direction: forward-left, backward_left ...
        for direction in possible_directions:
            # s: (-1, 1), (-2, 2), (-3, 3) ...
            step_list = self.possible_step_directions[direction]

            for s in step_list:
                if np.all(np.sign(s) == np.sign(step)):
                    steps_in_direction.append(s)

        # sort ascending
        steps_in_direction = np.array(steps_in_direction)
        steps_in_direction = steps_in_direction[np.argsort([np.linalg.norm(item) for item in steps_in_direction])]
        steps_in_direction = steps_in_direction.tolist()
        steps_in_direction = [np.array(elem) for elem in steps_in_direction]
        return steps_in_direction

    def get_positions_in_direction(self, position):
        """returns positions (only the ones on the board!) in the direction of a given position, sorted ascending
        returns empty list if no positions in that direction
        """
        positions_in_direction = []
        # calculate direction in terms of an imagined step
        imaginary_step = position - self.position

        # steps in direction is sorted from smallest step to largest
        steps_in_direction = self._get_steps_in_direction(imaginary_step)

        for step in steps_in_direction:
            try:
                new_position = self.position + step
                cell_content = self.board.cells[tuple(new_position)]
                positions_in_direction.append(new_position)
            except KeyError:
                # out of bounds
                pass
        return positions_in_direction

    def convert_steps_to_positions(self):
        """returns all positions for the current piece, including unavailable ones
        (uses the piece's natural step directions and its current position)
        """
        new_positions = []

        # direction is forward_right, backward_right ...
        possible_directions = list(self.possible_step_directions.keys())
        for direction in possible_directions:
            for step in self.possible_step_directions[direction]:
                new_position = self.position + step
                # on the board
                try:
                    cell_content = self.board.cells[tuple(new_position)]
                    new_positions.append(new_position)

                except KeyError:
                    # out-of-bound step
                    pass

        return new_positions

    def get_closer_positions_in_direction(self, position):
        """returns positions which are in the direction of position but are closer-equal (inclusive) to us
        only considers positions which are on the board
        returns an empty list if there are no positions in that direction
        """
        closer_positions = []
        positions_in_direction = self.get_positions_in_direction(position)
        new_position_distance = np.linalg.norm(position - self.position)

        if positions_in_direction:
            for pos in positions_in_direction:
                if np.linalg.norm(pos - self.position) <= new_position_distance:
                    closer_positions.append(pos)

            # sort ascending
            closer_positions = np.array(closer_positions)
            closer_positions = closer_positions[
                np.argsort([np.linalg.norm(item - self.position) for item in closer_positions])]
            closer_positions = closer_positions.tolist()
            closer_positions = [np.array(elem) for elem in closer_positions]
            return closer_positions
        else:
            return closer_positions

    def get_legal_positions(self):
        """returns currently available positions (to which we can actually move)
        if a position is not returned by this function the chess bot won't be able to step there
         """

        legal_moves = []
        check_threat = []

        # our king is in check
        if self.player.king.king_in_check():

            # we are the king so we have to step out of check or capture
            if self.player.king == self:
                positions_in_check = self.get_positions_in_check()
                positions_in_check = [tuple(item) for item in positions_in_check]
                possible_positions = self.convert_steps_to_positions()

                for pos in possible_positions:
                    # if a position is not in check
                    if tuple(pos) not in positions_in_check:
                        # if it is empty -- legal_move
                        if self.board.cells[tuple(pos)] is None:
                            legal_moves.append(pos)
                        # if it has an opponent on it -- legal_move
                        elif self.board.cells[tuple(pos)].player == self.opponent:
                            legal_moves.append(pos)
                # see if we can capture it
                piece_causing_check = self.opponent.piece_giving_check()
                if tuple(piece_causing_check.position) in [tuple(item) for item in possible_positions]:
                    legal_moves.append(piece_causing_check.position)

                legal_moves = self.remove_duplicate_positions(legal_moves)
                return legal_moves

            # we are not the king
            else:
                # if we are blocking another check -- we can't move
                for opp_piece in self.opponent.pieces:
                    if self.blocking_check(from_piece=opp_piece):
                        return legal_moves

                # if we are not blocking another check -- we can move in between the attacker and the king or capture
                possible_positions = self.convert_steps_to_positions()
                possible_positions = [tuple(item) for item in possible_positions]
                piece_causing_check = self.opponent.piece_giving_check()
                # see if we can capture it
                if tuple(piece_causing_check.position) in possible_positions:

                    if self.piece_type == "pawn":
                        capture_right_pos = tuple(self.position + self.possible_step_directions["capture_right"][0])
                        capture_left_pos = tuple(self.position + self.possible_step_directions["capture_left"][0])

                        if tuple(piece_causing_check.position) in (capture_right_pos, capture_left_pos):
                            legal_moves.append(piece_causing_check.position)

                    elif self.piece_type == "knight":
                        if tuple(piece_causing_check.position) in possible_positions:
                            legal_moves.append(piece_causing_check.position)
                    else:
                        closer_positions = self.get_closer_positions_in_direction(piece_causing_check.position)
                        # strictly closer is empty
                        if sum([0 if self.board.cells[tuple(pos)] is None else 1 for pos in
                                  closer_positions][:-1]) == 0:
                            legal_moves.append(piece_causing_check.position)

                # if it is a rook, queen, bishop -- we can move in between
                if piece_causing_check.piece_type in ("rook", "queen", "bishop"):
                    in_between_positions = piece_causing_check.get_closer_positions_in_direction(
                        self.player.king.position)

                    #  in_between_positions is not an empty list
                    if in_between_positions:
                        in_between_positions = [tuple(item) for item in in_between_positions]

                        # pawn
                        if self.piece_type == "pawn":
                            capture_right_pos = tuple(
                                self.position + self.possible_step_directions["capture_right"][0])
                            capture_left_pos = tuple(self.position + self.possible_step_directions["capture_left"][0])

                            for pos in possible_positions:
                                if tuple(pos) in in_between_positions:
                                    if tuple(pos) not in (capture_right_pos, capture_left_pos):
                                        legal_moves.append(pos)

                        # knight
                        elif self.piece_type == "knight":
                            for pos in possible_positions:
                                if tuple(pos) in in_between_positions:
                                    legal_moves.append(pos)

                        else:

                            for pos in possible_positions:
                                if tuple(pos) in in_between_positions:

                                    closer_positions = self.get_closer_positions_in_direction(pos)
                                    count_non_empty = \
                                        sum([0 if self.board.cells[tuple(p)] is None else 1 for p in closer_positions])
                                    if count_non_empty == 0:
                                        legal_moves.append(piece_causing_check.position)
                                    elif count_non_empty == 1:
                                        if tuple(pos) == tuple(piece_causing_check.position):
                                            legal_moves.append(pos)

                    # in_between_positions is an empty list []
                    else:
                        pass

                legal_moves = self.remove_duplicate_positions(legal_moves)
                return legal_moves

        # our king is not in check
        else:
            possible_positions = self.convert_steps_to_positions()
            possible_positions = self.remove_duplicate_positions(possible_positions)

            # if we are king
            if self.piece_type == "king":
                positions_in_check = self.get_positions_in_check()
                positions_in_check = [tuple(item) for item in positions_in_check]

                for pos in possible_positions:

                    # the position is not in check
                    if tuple(pos) not in positions_in_check:

                        # position is empty
                        if self.board.cells[tuple(pos)] is None:
                            legal_moves.append(pos)
                            check_threat.append(pos)

                        # opponent's piece
                        elif self.board.cells[tuple(pos)].player is self.opponent:
                            legal_moves.append(pos)
                            check_threat.append(pos)

                        # our piece
                        elif self.board.cells[tuple(pos)].player is self.player:
                            check_threat.append(pos)
                        else:
                            raise ValueError("Something does not work with if, elif, elif")

            # we are not king
            else:

                for new_position in possible_positions:
                    cell_content = self.board.cells[tuple(new_position)]
                    # cell is empty
                    if cell_content is None:

                        # pawn
                        if self.piece_type == "pawn":
                            capture_right_pos = tuple(self.position + self.possible_step_directions["capture_right"][0])
                            capture_left_pos = tuple(self.position + self.possible_step_directions["capture_left"][0])


                            # we are not trying to capture an empty cell
                            if tuple(new_position) not in (capture_right_pos, capture_left_pos):
                                legal_moves.append(new_position)

                        # knight
                        elif self.piece_type == "knight":
                            check_threat.append(new_position)
                            legal_moves.append(new_position)

                        # rook, queen, bishop
                        else:
                            # if all closer positions to us are also empty
                            closer_positions = self.get_closer_positions_in_direction(new_position)

                            # closer positions is a list but all cells are empty
                            if closer_positions:
                                if sum([0 if self.board.cells[tuple(pos)] is None else 1 for pos in
                                        closer_positions]) == 0:
                                    legal_moves.append(new_position)

                            # strictly closer is empty
                            elif sum([0 if self.board.cells[tuple(pos)] is None else 1 for pos in
                                      closer_positions][:-1]) == 0:

                                # equal is not empty
                                if [0 if self.board.cells[tuple(pos)] is None else 1 for pos in
                                    closer_positions][-1] == 1:
                                    legal_moves.append(new_position)

                    # opponent's piece
                    elif cell_content.player is self.opponent:

                        # pawn
                        if self.piece_type == "pawn":
                            capture_right_pos = tuple(self.position + self.possible_step_directions["capture_right"][0])
                            capture_left_pos = tuple(self.position + self.possible_step_directions["capture_left"][0])

                            # if new_pos is not straight ahead -- legal move and check threat
                            if tuple(new_position) in (capture_right_pos, capture_left_pos):
                                legal_moves.append(new_position)

                        # knight
                        elif self.piece_type == "knight":

                            legal_moves.append(new_position)

                        # rook, queen, bishop
                        else:
                            closer_positions = self.get_closer_positions_in_direction(new_position)

                            # closer positions is a list
                            if closer_positions:
                                #  if all closer positions to us are empty
                                if sum([0 if self.board.cells[tuple(pos)] is None else 1 for pos in
                                        closer_positions]) == 0:

                                    legal_moves.append(new_position)

                                # strictly closer is empty
                                elif sum([0 if self.board.cells[tuple(pos)] is None else 1 for pos in
                                          closer_positions][:-1]) == 0:

                                    # equal is not empty
                                    if [0 if self.board.cells[tuple(pos)] is None else 1 for pos in
                                        closer_positions][-1] == 1:
                                        legal_moves.append(new_position)

                                # not all closer positions are empty
                                else:
                                    pass

                    # own piece standing there
                    else:
                        pass

                    # See if we are blocking check
                    # blocking check already considers that we are the only ones blocking the king from them
                    for opp_piece in self.opponent.pieces:

                        # if check is given by rook, queen, bishop -- we can step closer or remove
                        if self.blocking_check(from_piece=opp_piece):
                            possible_positions = self.convert_steps_to_positions()
                            possible_positions = self.remove_duplicate_positions(possible_positions)
                            legal_moves = []    # empty legal_moves since we can only move this way now

                            for new_position in possible_positions:
                                # pawn
                                if self.piece_type == "pawn":

                                    capture_right_pos = tuple(self.position + self.possible_step_directions["capture_right"][0])
                                    capture_left_pos = tuple(self.position + self.possible_step_directions["capture_left"][0])

                                    # if new_pos is not straight ahead -- legal move and check threat
                                    if tuple(new_position) in (capture_right_pos, capture_left_pos):
                                        if tuple(opp_piece.position) == tuple(new_position):
                                            legal_moves.append(new_position)
                                    else:
                                        in_between_positions = self.get_closer_positions_in_direction(opp_piece.position)

                                        in_between_positions = [tuple(item) for item in in_between_positions]
                                        if tuple(new_position) in in_between_positions:
                                            if self.board.cells[tuple(new_position)] is None:
                                                legal_moves.append(new_position)

                                # rook, queen, bishop, knight
                                else:
                                    positions_in_direction_of_check = \
                                        self.get_closer_positions_in_direction(opp_piece.position)
                                    positions_in_direction_of_check = [tuple(item) for item in
                                                                       positions_in_direction_of_check]

                                    if tuple(new_position) in positions_in_direction_of_check:
                                        legal_moves.append(new_position)
                        else:
                            # we are not blocking check from that opp_piece
                            pass

        legal_moves = self.remove_duplicate_positions(legal_moves)
        return legal_moves


class Pawn(Piece):
    def set_step_directions(self):
        self.white_steps = {
            'normal_step': [np.array([0, 1])],
            'capture_right': [np.array([1, 1])],
            'capture_left': [np.array([-1, 1])],
            'initial_long_step': [np.array([0, 2])]
        }
        self.black_steps = {
            'normal_step': [np.array([0, -1])],
            'capture_right': [np.array([-1, -1])],
            'capture_left': [np.array([1, -1])],
            'initial_long_step': [np.array([0, -2])]
        }
        self.possible_step_directions = self.white_steps if self.color == "white" else self.black_steps

    def promote(self, new_position, verbose=False):
        active_player = self.board.player_1 if self.color == self.board.player_1.color else self.board.player_2
        if self.color == "white":
            if tuple(new_position) in [(1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8)]:
                pass
            else:
                return 0

        else:
            if tuple(new_position) in [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]:
                pass
            else:
                return 0

        active_player_picece_types = [p.piece_type for p in active_player.pieces]
        if active_player_picece_types.count("queen") == 0:
            active_player.pop_piece(self, verbose)
            active_player.pieces.append(Queen(name="queen_new", color=active_player.color))

        elif active_player_picece_types.count("rook") in (0, 1):
            active_player.pop_piece(self, verbose)
            active_player.pieces.append(Rook(name="rook_new", color=active_player.color))

        elif active_player_picece_types.count("bishop") in (0, 1):
            active_player.pop_piece(self, verbose)
            active_player.pieces.append(Bishop(name="bishop_new", color=active_player.color))

        elif active_player_picece_types.count("knight") in (0, 1):
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
                    p.player = active_player
                    p.opponent = active_player.opponent

    def convert_steps_to_positions(self):
        """returns all positions for the current piece, including unavailable ones"""
        available_new_positions = []
        # We can move forward ONE cell if that cell is empty and we are not blocking a check currently
        try:
            if self.board.cells[tuple(self.position + self.possible_step_directions['normal_step'][0])] is None:
                for opp_piece in self.opponent.pieces:
                    if not self.blocking_check(from_piece=opp_piece):
                        available_new_positions.append(self.position + self.possible_step_directions['normal_step'][0])
        except KeyError:
            pass

        # We can capture RIGHT if the opponents piece is there and we are not blocking check currently
        try:
            if self.board.cells[tuple(self.position + self.possible_step_directions['capture_right'][0])] is not None:
                if self.board.cells[
                    tuple(self.position + self.possible_step_directions['capture_right'][0])].color != self.color:
                    for opp_piece in self.opponent.pieces:
                        if not self.blocking_check(from_piece=opp_piece):
                            available_new_positions.append(
                                self.position + self.possible_step_directions['capture_right'][0])
        except KeyError:
            pass

        # We can capture LEFT if the opponents piece is there and we are not blocking check currently
        try:
            if self.board.cells[tuple(self.position + self.possible_step_directions['capture_left'][0])] is not None:
                if self.board.cells[
                    tuple(self.position + self.possible_step_directions['capture_left'][0])].color != self.color:
                    for opp_piece in self.opponent.pieces:
                        if not self.blocking_check(from_piece=opp_piece):
                            available_new_positions.append(
                                self.position + self.possible_step_directions['capture_left'][0])
        except KeyError:
            pass

        # We can move forward TWO cells if that cell is empty, we haven't moved yet and we are not blocking a check currently
        try:
            if self.board.cells[tuple(self.position + self.possible_step_directions['initial_long_step'][0])] is None:
                if self.board.cells[tuple(self.position + self.possible_step_directions['normal_step'][0])] is None:
                    if np.all(self.position == self.starting_position):
                        for opp_piece in self.opponent.pieces:
                            if not self.blocking_check(from_piece=opp_piece):
                                available_new_positions.append(
                                    self.position + self.possible_step_directions['initial_long_step'][0])
        except KeyError:
            pass
        return available_new_positions


class King(Piece):
    def set_step_directions(self):
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

    def convert_steps_to_positions(self):
        """converts steps to positions using: current_position + step for step in possible_steps"""
        # castling not implemented yet !!

        new_positions = []
        step_directions = list(self.possible_step_directions.values())

        for direction in step_directions:
            new_position = self.position + direction
            # on the board
            try:
                cell_content = self.board.cells[tuple(new_position)]
                new_positions.append(new_position)
            except KeyError:
                # out-of-bound step
                pass

        return new_positions

    def king_in_check(self):
        """returns True if we are in check, else False"""
        positions_in_check = self.get_positions_in_check()
        positions_in_check = [tuple(item) for item in positions_in_check]
        if tuple(self.position) in positions_in_check:
            return True
        else:
            return False

    def other_king_distance(self, from_position):
        """returns the distance from the other player's king
        args:
        from_position: np.array, opponent's king's position
        """
        opponent_king = None

        for p in self.opponent.pieces:
            if p.piece_type == self.piece_type:
                opponent_king = p

        # if opponent king remains None I will get an error
        distance = np.linalg.norm(opponent_king.position - from_position)
        return distance


class Bishop(Piece):
    def set_step_directions(self):
        self.white_steps = {
            'forward_right': [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4]),
                              np.array([5, 5]), np.array([6, 6]), np.array([7, 7])],

            'forward_left': [np.array([-1, 1]), np.array([-2, 2]), np.array([-3, 3]), np.array([-4, 4]),
                             np.array([-5, 5]), np.array([-6, 6]), np.array([-7, 7])],

            'backward_right': [np.array([1, -1]), np.array([2, -2]), np.array([3, -3]), np.array([4, -4]),
                               np.array([5, -5]), np.array([6, -6]), np.array([7, -7])],

            'backward_left': [np.array([-1, -1]), np.array([-2, -2]), np.array([-3, -3]), np.array([-4, -4]),
                              np.array([-5, -5]), np.array([-6, -6]), np.array([-7, -7])]
        }
        self.black_steps = {
            'forward_right': [np.array([-1, -1]), np.array([-2, -2]), np.array([-3, -3]), np.array([-4, -4]),
                              np.array([-5, -5]), np.array([-6, -6]), np.array([-7, -7])],

            'forward_left': [np.array([1, -1]), np.array([2, -2]), np.array([3, -3]), np.array([4, -4]),
                             np.array([5, -5]), np.array([6, -6]), np.array([7, -7])],

            'backward_right': [np.array([-1, 1]), np.array([-2, 2]), np.array([-3, 3]), np.array([-4, 4]),
                               np.array([-5, 5]), np.array([-6, 6]), np.array([-7, 7])],

            'backward_left': [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4]),
                              np.array([5, 5]), np.array([6, 6]), np.array([7, 7])],
        }

        self.possible_step_directions = self.white_steps if self.color == "white" else self.black_steps


class Rook(Piece):
    def set_step_directions(self):
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
    def set_step_directions(self):
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
            'backward_left': [np.array([-1, -1]), np.array([-2, -2]), np.array([-3, -3]), np.array([-4, -4]),
                              np.array([-5, -5]), np.array([-6, -6]), np.array([-7, -7])]
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
            'forward_right': [np.array([-1, -1]), np.array([-2, -2]), np.array([-3, -3]), np.array([-4, -4]),
                              np.array([-5, -5]), np.array([-6, -6]), np.array([-7, -7])],
            'forward_left': [np.array([1, -1]), np.array([2, -2]), np.array([3, -3]), np.array([4, -4]),
                             np.array([5, -5]), np.array([6, -6]), np.array([7, -7])],
            'backward_right': [np.array([-1, 1]), np.array([-2, 2]), np.array([-3, 3]), np.array([-4, 4]),
                               np.array([-5, 5]), np.array([-6, 6]), np.array([-7, 7])],
            'backward_left': [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4]),
                              np.array([5, 5]), np.array([6, 6]), np.array([7, 7])]
        }

        self.possible_step_directions = self.white_steps if self.color == "white" else self.black_steps


class Knight(Piece):
    def set_step_directions(self):
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

    def convert_steps_to_positions(self):
        """returns all positions for the current piece, including unavailable ones"""

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

        return available_new_positions


class Player:
    def __init__(self, color, engine, id=None, max_depth=None, model_path=None, kernel_initializer='glorot_uniform'):
        self.color = color
        self.engine = engine
        self.id = id
        self.elo_score = 1400
        self.opponent = None
        self.pawns = [Pawn(name=pawn_name, color=self.color) for pawn_name in PAWN_NAMES]
        self.king = King(name=KING_NAME, color=self.color)
        self.queen = Queen(name=QUEEN_NAME, color=self.color)
        self.bishops = [Bishop(name=bishop_name, color=self.color) for bishop_name in BISHOP_NAMES]
        self.rooks = [Rook(name=rook_name, color=self.color) for rook_name in ROOK_NAMES]
        self.knights = [Knight(name=knight_name, color=self.color) for knight_name in KNIGHT_NAMES]
        self.pieces = self.pawns + self.bishops + self.knights + self.rooks + [self.king] + [self.queen]
        self.active = False
        self.board = None
        self.steps_encoded = self.encode_all_steps()

        # Verifying other arguments of engine
        if self.engine not in (0, 1, 2, "human", "robot", "ai"):
            raise ValueError("engine should be one of: 0,1,2,human, robot, ai")
        else:
            # Robot: forward looking
            if self.engine == "robot" or self.engine == 1:
                if max_depth not in (0, 1, 2, "adaptive"):
                    raise ValueError("Invalid max_depth for {self.color}. Choose one of: 0,1,2,'adaptive'.")
                else:
                    self.max_depth = max_depth

            # AI: neural net
            elif self.engine == "ai" or self.engine == 2:
                if model_path:
                    with open(model_path, "rb") as pickle_file:
                        self.nn = pickle.load(pickle_file)
                        self.nn.player = self
                        self.model_path = model_path
                else:
                    # No model path given -- initialize a new NeuralNet
                    print("No model path: Initializing a new NeuralNet")
                    self.nn = NeuralNet(kernel_initializer=kernel_initializer)
                    self.nn.player = self


    def __repr__(self):
        return self.color

    def encode_all_steps(self):
        """creates an encoding of (x,y,id) for all positions on the board combined with all piece.id in self.pieces"""
        ids = [p.id for p in self.pieces]
        all_combo = list(itertools.product(range(1, 9), range(1, 9), ids))
        all_steps_encoded = [int(str(item[0]) + str(item[1]) + str(item[2])) for item in all_combo]
        return sorted(all_steps_encoded)

    def encode_legal_steps(self):
        """creates an encoding of (x,y,id) for legal positions combined with piece.id in available pieces"""
        legal_steps_encoded = []
        for p in self.pieces:
            legal_pos = p.get_legal_positions()
            if legal_pos:
                for pos in legal_pos:
                    step_encoded = int(str(pos[0]) + str(pos[1]) + str(p.id))
                    legal_steps_encoded.append(step_encoded)

        return legal_steps_encoded

    def decode_step(self, encoded_step):
        s = str(encoded_step)
        x, y, id = int(s[0]), int(s[1]), int(s[2:])
        return x, y, id

    def pop_piece(self, piece, verbose=False):
        """removes a piece from Player.pieces (used when the piece is captured)"""
        if piece.piece_type != 'king':
            popped = self.pieces.pop(self.pieces.index(piece))
            if verbose:
                print(popped)
        else:
            print(self.board)
            raise ValueError("Tried to remove king")

    def get_available_pieces(self):
        """returns a sublist of Player.pieces for which legal steps are available"""
        available = []
        for piece in self.pieces:
            avail = piece.get_legal_positions()
            if avail:
                available.append(piece)
        return available

    def get_piece(self, position=None, id=None, name=None):
        """returns a Piece instance based on either position or id"""
        if position is not None:
            for piece in self.pieces:
                if np.all(piece.position == position):
                    return piece
            return None
        elif id is not None:
            for piece in self.pieces:
                if piece.id == id:
                    return piece
            return None
        elif name is not None:
            for piece in self.pieces:
                if piece.name == name or piece.__repr__() == name:
                    return piece
            return None

    def calculate_score(self):
        """ Chess bot only:
        get the score of the player at a particular point during the game
        this is used to rank steps for the chess bot
        """
        our_total = sum([piece.value for piece in self.pieces])
        opponent_total = sum([piece.value for piece in self.opponent.pieces])

        if self.king.king_in_check():
            our_total -= 100
            if self.losing:
                our_total -= 1000

        if self.opponent.king.king_in_check():
            our_total += 100
            if self.opponent.losing:
                our_total += 1000

        our_score = (our_total - opponent_total) + random.randint(1, 3)
        return our_score

    def look_forward(self, counter=0, max_depth=1):
        """ Chess bot only:
        Looks ahead max_depth steps and decides which step to take based on the outcome of board.calculate_score
        returns current_best = [score, piece, new_position]"""

        if counter == max_depth:
            current_best_score = -100000000
            current_best_move = None

            available_pieces = self.get_available_pieces()
            for piece in available_pieces:

                avail_positions = piece.get_legal_positions()
                avail_positions = self.king.remove_duplicate_positions(avail_positions)

                for pos in avail_positions:

                    piece.move(to=pos)

                    if counter == max_depth:
                        score = self.calculate_score() # HAVE TO IMPLEMENT THIS !!!!
                        # print("score", score)
                        self.board.pop_last_move()

                        if score > current_best_score:
                            current_best_score = score
                            current_best_move = (piece, pos)

            if max_depth == 0:
                return current_best_move
            else:
                return current_best_score

        else:
            opponents_max = 100000000
            current_best_move = None
            available_pieces = self.get_available_pieces()

            for piece in available_pieces:
                avail_positions = piece.get_legal_positions()
                avail_positions = self.king.remove_duplicate_positions(avail_positions)

                for pos in avail_positions:
                    piece.move(to=pos)
                    opponents_best_score = self.opponent.look_forward(counter=(counter+1), max_depth=0)
                    self.board.pop_last_move()

                    if opponents_best_score < opponents_max:
                        opponents_max = opponents_best_score
                        current_best_move = (piece, pos)

            if counter == 0:
                return current_best_move
            else:
                return opponents_max

    def select_depth(self, verbose=True):
        avail_pieces = self.get_available_pieces()
        # opp_avail_pieces = self.opponent.get_available_pieces()
        complexity = []
        for p in avail_pieces:
            complexity.extend(p.get_legal_positions())
        # for opp_p in opp_avail_pieces:
        #     complexity.extend(opp_p.get_legal_positions())

        if len(complexity) < 4:
            d = 2
        elif len(complexity) < 8:
            d = 1
        elif len(complexity) < 20:
            d = 0
        else:
            d = 0
        if verbose:
            print(self.color, "depth: ", d)
        return d

    def encode_board(self):
        """Creates an encoding for the current board where each cell has the id of the piece standing on it
        returns a 64-long vector with numbers from 0-32
        """
        encoded_board = []
        for pos in self.board.all_positions:
            try:
                encoded_board.append(self.board.cells[pos].id)
            except AttributeError:
                encoded_board.append(0)

        return encoded_board

    def human_move(self):
        print(" ")
        print("Your turn.")
        user_input = input("Select: piece, x, y \n")
        user_input = user_input.split(sep=",")
        piece_repr = user_input[0]
        x = int(user_input[1])
        y = int(user_input[2])

        # pop spaces from front and back
        if piece_repr[-1] == " ":
            piece_repr.pop(-1)
        elif piece_repr[0] == " ":
            piece_repr.pop(0)

        p = self.get_piece(name=piece_repr)
        new_pos = np.array([x, y])

        legal = p.get_legal_positions()
        if legal:
            if tuple(new_pos) in [tuple(item) for item in legal]:
                return p, new_pos
            else:
                print("Invalid step!")
                p, new_pos = self.human_move()
                return p, new_pos
        else:
            print("Invalid piece!")
            p, new_pos = self.human_move()
            return p, new_pos


    def choose_move(self):
        if self.engine == "human" or self.engine == 0:
            p, new_pos = self.human_move()
        elif self.engine == "robot" or self.engine == 1:
            if self.max_depth == "adaptive":
                depth = self.select_depth()
            else:
                depth = self.max_depth
            p, new_pos = self.look_forward(max_depth=depth)
        else:
            current_board = self.encode_board()
            p, new_pos = self.nn.forward_pass(current_board)

        if not isinstance(p, Piece):
            print("Possible tie")
        return p, new_pos

    def piece_giving_check(self):
        """returns our player's pieces which are giving check to opponent's king"""
        for p in self.pieces:
            if p.giving_check():
                return p

    @property
    def losing(self):
        """returns True if we are suffering check mate, ends the game"""
        if self.king.king_in_check():
            check_giving_piece = self.opponent.piece_giving_check()

            if not isinstance(check_giving_piece, Piece):
                return False

            for p in self.pieces:
                if not p.capture_ability(check_giving_piece):
                    avail = self.king.get_legal_positions()
                    if avail:
                        return False
                    else:
                        if check_giving_piece.piece_type != "knight":
                            step_between_them = []
                            check_threat = check_giving_piece.positions_kept_in_check()
                            direction_signs = np.sign(self.king.position - check_giving_piece.position)

                            for pos in check_threat:
                                if np.all(np.sign(pos - check_giving_piece.position) == direction_signs):
                                    step_between_them.append(pos)

                            avail_pieces = self.get_available_pieces()

                            for piece in avail_pieces:
                                avail_positions = piece.get_legal_positions()
                                for pos in step_between_them:
                                    if tuple(pos) in [tuple(item) for item in avail_positions]:
                                        return False
                else:
                    return False

            # After the for loop
            # None of the pieces returned False -- so it must be True
            return True
        else:
            return False


class Board:
    def __init__(self, player_1, player_2, reset=False, max_steps=1000):
        self.all_positions = list(itertools.product(range(1, 9), range(1, 9)))
        self.player_1 = player_1
        self.player_2 = player_2
        self.players = [player_1, player_2]
        self.max_steps = max_steps
        self.move_history = []
        if reset:
            self.reset()
        # set board
        self.set_board_for_players_and_pieces()
        # set opponents for both players
        self.set_opponents_for_players_and_pieces()
        # set players for each piece
        self.set_players_for_pieces()
        # initialize all cells with None
        self.cells = {pos: None for pos in self.all_positions}
        # set each cell's content to the corresponding piece if it is occupied by one, else leave it None
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

    def reset(self):
        for player in self.players:
            player.pawns = [Pawn(name=pawn_name, color=player.color) for pawn_name in PAWN_NAMES]
            player.king = King(name=KING_NAME, color=player.color)
            player.queen = Queen(name=QUEEN_NAME, color=player.color)
            player.bishops = [Bishop(name=bishop_name, color=player.color) for bishop_name in BISHOP_NAMES]
            player.rooks = [Rook(name=rook_name, color=player.color) for rook_name in ROOK_NAMES]
            player.knights = [Knight(name=knight_name, color=player.color) for knight_name in KNIGHT_NAMES]
            player.pieces = player.pawns + player.bishops + player.knights + player.rooks + [player.king] + [player.queen]

    def update_elo_score(self, winner, loser, tie, capture_bias=False):
        k_factor = 32
        base = 10
        if not tie:
            assert (winner is not None) and (loser is not None)
            actual_winner = 1
            actual_loser = 0
        else:
            assert (winner is None) and (loser is None)
            winner = self.player_1
            loser = self.player_2
            actual_winner = 0.5
            actual_loser = 0.5

        expected_winner = 1 / (1 + base**((loser.elo_score - winner.elo_score)/400))
        expected_loser = 1 / (1 + base**((winner.elo_score - loser.elo_score)/400))
        winner.elo_score = winner.elo_score + (k_factor*(actual_winner - expected_winner))
        loser.elo_score = loser.elo_score + (k_factor*(actual_loser - expected_loser))

        if capture_bias:
            # calculate the final value of their pieces at a tie and add to the elo score
            winner_piece_value = winner.calculate_score()
            loser_piece_value = loser.calculate_score()
            diff = winner_piece_value - loser_piece_value

            # if diff is positive add to winner, subtract from loser
            # if diff is negative add negative to winner, subtract negative from loser (actually played better so winner)
            winner.elo_score = winner.elo_score + (0.1*diff)
            loser.elo_score = loser.elo_score - (0.1*diff)
            print("Elo scores: ", winner.elo_score, loser.elo_score)

    def pop_last_move(self):
        """ Chess bot only:
        deletes the last move from self.move_history"""
        piece, old_pos, new_pos = self.move_history[-1]
        piece.position = old_pos
        self.move_history.pop(-1)

    def set_opponents_for_players_and_pieces(self):
        self.player_1.opponent = self.player_2
        for piece in self.player_1.pieces:
            piece.opponent = self.player_1.opponent

        self.player_2.opponent = self.player_1
        for piece in self.player_2.pieces:
            piece.opponent = self.player_2.opponent

    def set_players_for_pieces(self):
        for piece in self.player_1.pieces:
            piece.player = self.player_1
        for piece in self.player_2.pieces:
            piece.player = self.player_2

    def set_board_for_players_and_pieces(self):
        for player in self.players:
            player.board = self
            for piece in player.pieces:
                piece.board = self

    def get_player(self, color):
        """returns a Player instance based on its color"""
        if self.player_1.color == color:
            return self.player_1
        elif self.player_2.color == color:
            return self.player_2
        else:
            raise ValueError(f"no player with color: {color}")

    def update_board(self):
        """Update every cell to contain a piece object based on the piece's position"""
        self.cells = {pos: None for pos in self.all_positions}
        for player in self.players:
            for piece in player.pieces:
                self.cells[tuple(piece.position)] = piece

    def play(self, show=True, verbose=True):
        """Plays one chess game

        args:
        show: bool, if True prints the board after every single step
        verbose: bool, if True prints every moved piece and its new position

        The 'black' chess bot adaptively selects the max_depth to which it looks ahead
                based on the number of legal steps it can take (e.g. advantageous when its king is in check)
        The 'white' chess bot consistently looks 1 step ahead (which is defined as max_depth=0)
        """
        winner = None
        loser = None
        tie = False

        i = 0
        run = True
        while run:
            if i % 2 == 0:
                active_player = self.player_1
                passive_player = self.player_2
            else:
                active_player = self.player_2
                passive_player = self.player_1

            active_piece, new_pos = active_player.choose_move()

            if verbose:
                print(". ")
                print(". ")
                print(". ")
                print(". ")
                print(". ")
                print(". ")
                print(". ")
                print(". ")
                print(f"{active_player.engine.upper()}'s move: {active_piece} to {tuple(new_pos)}")

            if active_piece:
                # if opponent is there -- remove that piece from the board
                capture = passive_player.get_piece(new_pos)
                if capture is not None:
                    passive_player.pop_piece(capture, verbose)
                active_piece.move(to=new_pos)
                active_piece.promote(new_pos, verbose)

            else:
                if run:
                    print("Finished with a tie!")
                    tie = True
                    return winner, loser, tie

            self.update_board()

            if show:
                # show board
                print(self)
                time.sleep(3)

            # increment i
            i += 1
            if (i == 50) or (i == 100) or (i == 150):
                print(i)

            # Check if anyone is losing
            run = not np.any([self.player_1.losing, self.player_2.losing])

            # Check if we only have king vs king
            if (len(self.player_1.pieces) + len(self.player_2.pieces)) == 2:
                print("King vs King -- stopping game")
                run = False

            # Check if we reached 1000 steps
            if i > self.max_steps:
                print(f"Reached max_steps={self.max_steps} -- stopping game")
                run = False

            if not run:
                print(self)
                if self.player_1.losing:
                    print(f"{self.player_1.color} has lost")
                    winner = self.player_2
                    loser = self.player_1
                elif self.player_2.losing:
                    print(f"{self.player_2.color} has lost")
                    winner = self.player_1
                    loser = self.player_2
                else:
                    tie = True
                    return winner, loser, tie
                print("Total number of steps: ", i)
        return winner, loser, tie

def main():
    """Initialize and play one game with two chess bots against each other"""

    player_1 = Player("white", engine="human")
    player_2 = Player("black", engine="ai")

    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")



    print("Initializing game.")
    print("player_1 = Player('white', engine='human')")
    print("player_2 = Player('black', engine='ai')")
    print("board = Board(player_1, player_2")
    print("board.play(show=True)")

    board = Board(player_1, player_2)
    print(board)
    winner, loser, tie = board.play(show=True, verbose=True)



    print("winner", winner)
    print("loser", loser)
    print("tie", tie)



if __name__ == '__main__':
    main()
