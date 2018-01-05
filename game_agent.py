"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
#import hashlib
from math import hypot
from math import floor


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def proximity_score(game, player):
    """The "Proximity" evaluation function outputs a score equal to the
    difference between the maximum distance and the current distance
    separating the two players on a board. A higher score is returned for
    board states that put the active player closer to their opponent.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    player_location = game.get_player_location(player)
    opponent_location = game.get_player_location(game.get_opponent(player))
    distance = hypot(opponent_location[0] - player_location[0], opponent_location[1] - player_location[1])
    max_distance = hypot(game.width - 0, game.height - 0)
    return max_distance - distance


def improved_score(game, player):
    """The "Improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


def center_score(game, player):
    """Outputs a score equal to square of the distance from the center of the
    board to the position of the player.

    This heuristic is only used by the autograder for testing.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """

    player_location = game.get_player_location(player)
    center_location = floor(game.width / 2.), floor(game.height / 2.)
    distance = hypot(center_location[0] - player_location[0], center_location[1] - player_location[1])
    max_distance = hypot(center_location[0], center_location[1])
    return max_distance - distance / 100  # max distance from center = .98


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic function is designed to draw an opponent to the left side
    of the board. When the player is not in a position to draw the opponent,
    the pre-built “improved” heuristic is used to score the game state.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    opponent = game.get_opponent(player)
    player_moves = game.get_legal_moves(player)
    player_location = game.get_player_location(player)
    opponent_moves = game.get_legal_moves(opponent)
    opponent_location = game.get_player_location(opponent)

    if opponent_location is None:
        return center_score(game, player)

    _, opponent_x = opponent_location
    opponent_y, _ = opponent_location
    upper_left_target = (opponent_y - 2, opponent_x - 1)
    lower_left_target = (opponent_y + 2, opponent_x - 1)
    upper_player_target = (opponent_y - 1, opponent_x + 1)
    lower_player_target = (opponent_y + 1, opponent_x + 1)

    if game.is_loser(player):  # or player_move_count == 0
        return float("-inf")

    if game.is_winner(player):  # or opponent_move_count == 0:
        return float("inf")

    # check that the player is in position, and that at least one opponent target is a legal move
    if (player_location == upper_player_target or player_location == lower_player_target) \
            and (upper_left_target in opponent_moves or lower_left_target in player_moves):

        draw_left_score = 1000
    else:
        draw_left_score = improved_score(game, player) + center_score(game, player)

    return draw_left_score


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic function is the sum of proximity to the opponent and a
    weighted improvement of available next moves, current player minus opponent.
    Meaning, a higher score is returned for board states (proposed moves) that
    put the player closer to their opponent, while accounting for the available
    immediate next moves for both players afforded by each game state.

    An improved weight is used to lower the influence the available next moves
    advantage has on the overall score when combined with the proximity.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    improved_weight = .5
    improved = improved_score(game, player)
    weighted_improved = improved * improved_weight

    # (proximity_score(game, player) / game.move_count) + weighted_improved
    return weighted_improved + (proximity_score(game, player) / game.move_count)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic function is the sum of room to move and improved. The base
    score is the improved calculation provided in sample players. Higher scores
    are returned for moves into less crowded quadrants of the board.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    x = game.width / 2.
    y = game.height / 2.
    q1_spaces = []
    q2_spaces = []
    q3_spaces = []
    q4_spaces = []

    for space in game.get_blank_spaces():
        if space[0] < x:
            if space[1] < y:
                q1_spaces.append(space)
            else:
                q3_spaces.append(space)
        else:
            if space[1] < y:
                q2_spaces.append(space)
            else:
                q4_spaces.append(space)

    location = game.get_player_location(player)
    if location[0] < x:
        if location[1] < y:
            room_to_move = len(q1_spaces)
        else:
            room_to_move = len(q3_spaces)
    else:
        if location[1] < y:
            room_to_move = len(q2_spaces)
        else:
            room_to_move = len(q4_spaces)

    return improved_score(game, player) + (room_to_move / 2)


def custom_score_4(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic function is an implementation of Warnsdorf's rule, a
    heuristic for finding a knight's tour. The intuition behind using this in
    an adversarial search is that the player taking the longest available path
    will outlive their opponent.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    player_moves = game.get_legal_moves(player)
    player_move_count = len(player_moves)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    opponent_move_count = len(opponent_moves)

    if game.is_loser(player) or player_move_count == 0:
        return float("-inf")

    if game.is_winner(player) or opponent_move_count == 0:
        return float("inf")

    if player_move_count == 1 and player_moves[0] in opponent_moves:
        w_score = 0
    else:
        w_score = (1 / player_move_count)  # a value between zero and one

    return w_score


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=20.):
        super().__init__(search_depth, score_fn, timeout)

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        # print('MM', best_move)
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if game.is_loser(self):
            return -1, -1

        values = []
        # print(game.get_legal_moves(self))
        for m in game.get_legal_moves(self):
            values.append((self.min_score(game.forecast_move(m), depth - 1), m))
        return max(values, key=lambda x: x[0])[1]

    def min_score(self, game, depth):
        """ Return the value for a win (+1) if the game is over,
            otherwise utility based on search to the given depth.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if game.is_loser(self) or depth == 0:
            return self.score(game, self)

        values = [float("inf")]
        for move in game.get_legal_moves(game.get_opponent(self)):
            values.append(self.max_score(game.forecast_move(move), depth - 1))
        return min(values)

    def max_score(self, game, depth):
        """ Return the value for a win (+1) if the game is over,
            otherwise utility based on search to the given depth.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if game.is_loser(self) or depth == 0:
            return self.score(game, self)

        values = [float("-inf")]
        for move in game.get_legal_moves(self):
            values.append(self.min_score(game.forecast_move(move), depth - 1))
        return max(values)


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    transposition_table = {}

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        self.transposition_table = {}  # used only for sorting the root node scores

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        legal_moves = game.get_legal_moves(self)
        # print('legal_moves', legal_moves)

        if len(legal_moves) == 0:
            return best_move

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire. Iterative deepening
            # is used to call the search function with increasing values, and
            # capturing the best move values until SearchTimeout is raised.
            depth = 1
            while True:
                best_move = self.alphabeta(game, depth)
                # print('best_move @' + str(depth), best_move)
                depth += 1

        except SearchTimeout:
            # print('TIMEOUT', best_move)
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        # print('AB', best_move)
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if game.is_loser(self):
            return -1, -1

        # print(self.transposition_table)

        # timeout the iterative deepening when a winner has already been returned, or sort root moves
        legal_moves = game.get_legal_moves(self)
        if len(self.transposition_table) > 0 and \
                (any(value == float("inf") for value in self.transposition_table.values()) or
                 (len(self.transposition_table) == len(legal_moves) and
                  all(value == float("-inf") for value in self.transposition_table.values()))):
            # print('FOUND LEAF')
            raise SearchTimeout()  # prevents infinite iterative deepening

        elif len(self.transposition_table) > 0:
            sorted_moves = sorted(legal_moves, key=self.root_sort, reverse=True)
        else:
            sorted_moves = legal_moves

        values = []
        for move in sorted_moves:
            value = self.min_score(game.forecast_move(move), depth - 1, alpha, beta)
            values.append((value, move))
            self.transposition_table[move] = value
            alpha = max(alpha, max(values, key=lambda x: x[0])[0])
            if beta <= alpha:
                break  # β cut-off

        # print('returning best_move:', max(values, key=lambda x: x[0])[1], 'from', values)
        return max(values, key=lambda x: x[0])[1]

    def root_sort(self, move):
        if move in self.transposition_table:
            return self.transposition_table[move]
        else:
            return float("-inf")

    def min_score(self, game, depth, alpha, beta):
        """ Return the value for a win (+1) if the game is over,
            otherwise utility based on search to the given depth.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if game.is_loser(self) or depth == 0:
            return self.score(game, self)

        values = [float("inf")]
        for move in game.get_legal_moves(game.get_opponent(self)):
            values.append(self.max_score(game.forecast_move(move), depth - 1, alpha, beta))
            beta = min(beta, min(values))
            if beta <= alpha:
                break  # α cut-off
        return min(values)

    def max_score(self, game, depth, alpha, beta):
        """ Return the value for a win (+1) if the game is over,
            otherwise utility based on search to the given depth.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if game.is_loser(self) or depth == 0:
            return self.score(game, self)

        values = [float("-inf")]
        for move in game.get_legal_moves(self):
            values.append(self.min_score(game.forecast_move(move), depth - 1, alpha, beta))
            alpha = max(alpha, max(values))
            if beta <= alpha:
                break  # β cut-off
        return max(values)


if __name__ == "__main__":
    from isolation import Board

    # create an isolation board (by default 7x7)
    player1 = AlphaBetaPlayer()  # timeout at depth of about 7 using a 9x9 board
    player2 = MinimaxPlayer(search_depth=3, score_fn=improved_score)  # timeout at depth of about 5 using a 9x9 board
    game = Board(player1, player2, width=7, height=7)

    local_history = [[3, 3], [1, 0], [2, 1], [3, 1], [1, 3], [4, 3], [2, 5], [5, 5], [4, 6], [3, 4], [6, 5], [4, 2]
        , [4, 4], [2, 3], [3, 6], [1, 1], [2, 4], [3, 0], [4, 5], [2, 2], [5, 3], [4, 1], [6, 1], [6, 2]]
        #, [4, 0], [5, 4], [3, 2], [3, 5], [2, 0], [5, 6], [1, 2], [6, 4], [0, 4], [5, 2], [1, 6], [6, 0]]
    #for move in local_history:
    #    game.apply_move(move)

    """
    Losers...
    28c3b076ba1c8d110e0d4524f7e6791a
    
    """

    # players take turns moving on the board, so player1 should be next to move
    assert (player1 == game.active_player)
    board_state = str(game._board_state)
    #hash_state = hashlib.md5(board_state.encode()).hexdigest()
    #print("\nBoard hash:\n{!s}".format(hash_state))
    print(game.to_string())

    # play the remainder of the game automatically -- outcome can be "illegal
    # move", "timeout", or "forfeit"
    winner, history, outcome = game.play(time_limit=150)
    if winner == player1:
        winner_name = 'Player 1'
    else:
        winner_name = 'Player 2'
    print("\nWinner: {}, {}\nOutcome: {}".format(winner_name, winner, outcome))
    print(game.to_string())
    print("Move history:\n{!s}".format(history))
    # print("Game state:\n{!s}".format(game._board_state))
