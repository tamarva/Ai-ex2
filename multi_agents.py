import random
import sys
from enum import Enum, auto
from typing import Callable, Tuple

import numpy as np
import abc
import util
import math
from game import Agent, Action
from game_state import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score

        num_rows, num_cols = board.shape

        result = 0
        for row_index in range(num_rows - 1):
            for col_index in range(num_cols - 1):
                if col_index == num_cols - 1:
                    result += board[row_index, col_index]
                else:
                    result += abs(board[row_index, col_index] - board[row_index, col_index + 1])

        for col_index in range(num_cols - 1):
            for row_index in range(num_rows - 1):
                if row_index == num_rows - 1:
                    result += board[row_index, col_index]
                else:
                    result += abs(board[row_index, col_index] - board[row_index + 1, col_index])

        return -result


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class SearchAlgorithm(Enum):
    """The minimax algorithm."""
    MINIMAX = auto()

    """The alpha-beta algorithm."""
    ALPHABETA = auto()

    """The expectimax algorithm."""
    EXPECTIMAX = auto()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    """Represents the first player agent index."""
    FIRST_PLAYER = 0

    """Represents the second player agent index."""
    SECOND_PLAYER = 1

    evaluation_function: Callable[[GameState], int]
    depth: int
    algorithm: SearchAlgorithm

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.minimizing_function = None
        self.depth = int(depth)
        self.algorithm = None

    def get_action(self, game_state):
        """
        Implements the base entry for {@link get_action}. This method prepares the legal actions, gets the successors,
        and calls the required algorithm, based on the strategy.
        :param game_state: The current game state.
        :return: The best move, in perspective for the current agent, to perform.
        """

        # Did we got a valid depth value
        if self.depth < 1:
            return Action.STOP

        # Fetch the possible actions
        legal_actions = game_state.get_legal_actions(0)
        if not legal_actions:
            return Action.STOP

        # Generate the successors for each legal action
        successors = [game_state.generate_successor(0, action) for action in legal_actions]

        # Calculate the score for each legal action
        scores = np.array([self._min_player(successor, (self.depth * 2) - 1, MultiAgentSearchAgent.SECOND_PLAYER)
                           for successor in successors])
        print(
            f'Max Score : {scores.max()} ; Scores: {scores} :: Selected Move: {legal_actions[scores.argmax(axis=0)]} (idx={scores.argmax(axis=0)})')
        return legal_actions[scores.argmax(axis=0)]

    def _max_player(self, game_state: GameState, depth: int, agent_index: int, alpha=float('-inf'), beta=float('inf')):
        """
        Perform the maximizing player move.
        :param game_state: The current game state.
        :param depth: The depth of the search.
        :param agent_index: The player agent index.
        :param alpha: The alpha-value.
        :param beta: The beta-value.
        :return: The move score or best move.
        """
        # Iterate over the available actions
        if depth == 0:
            return self.evaluation_function(game_state)

        actions = game_state.get_legal_actions(agent_index)
        if not actions:
            return self.evaluation_function(game_state)

        next_player = MultiAgentSearchAgent.SECOND_PLAYER if agent_index == MultiAgentSearchAgent.FIRST_PLAYER \
            else MultiAgentSearchAgent.FIRST_PLAYER
        max_score = -math.inf
        for action in actions:
            successor = game_state.generate_successor(agent_index, action)
            max_score = max(max_score,
                            self._min_player(successor, depth - 1, next_player, alpha, beta))

            # Alpha-Beta cutoff
            if self.algorithm == SearchAlgorithm.ALPHABETA:
                if max_score > beta:
                    break
                alpha = max(alpha, max_score)

        return max_score

    def _min_player(self, game_state: GameState, depth: int, agent_index: int, alpha=float('-inf'), beta=float('inf')):
        """
        Perform the minimizing player move.
        :param game_state: The current game state.
        :param depth: The depth of the search.
        :param agent_index: The player agent index.
        :param alpha: The alpha-value.
        :param beta: The beta-value.
        :return: The move score or best move.
        """
        # Iterate over the available actions
        if depth == 0:
            return self.evaluation_function(game_state)

        actions = game_state.get_legal_actions(agent_index)
        if not actions:
            return self.evaluation_function(game_state)

        next_player = MultiAgentSearchAgent.SECOND_PLAYER if agent_index == MultiAgentSearchAgent.FIRST_PLAYER \
            else MultiAgentSearchAgent.FIRST_PLAYER
        min_score = +math.inf
        for action in actions:
            successor = game_state.generate_successor(agent_index, action)
            min_score = min(min_score, self._max_player(successor, depth - 1, next_player, alpha, beta))

            # Alpha-Beta cutoff
            if self.algorithm == SearchAlgorithm.ALPHABETA:
                if min_score < alpha:
                    break
                beta = min(beta, min_score)

        return min_score


class MinmaxAgent(MultiAgentSearchAgent):
    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        super().__init__(evaluation_function, depth)
        self.algorithm = SearchAlgorithm.MINIMAX


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        super().__init__(evaluation_function, depth)
        self.algorithm = SearchAlgorithm.ALPHABETA


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        super().__init__(evaluation_function, depth)
        self.algorithm = SearchAlgorithm.EXPECTIMAX

    def _min_player(self, game_state: GameState, depth: int, agent_index: int, alpha=float('-inf'), beta=float('inf')):
        """
        Perform the minimizing player move.
        :param game_state: The current game state.
        :param depth: The depth of the search.
        :param agent_index: The player agent index.
        :param alpha: The alpha-value.
        :param beta: The beta-value.
        :return: The move score or best move.
        """
        # Iterate over the available actions
        if depth == 0:
            return self.evaluation_function(game_state)

        actions = game_state.get_legal_actions(agent_index)
        if not actions:
            return self.evaluation_function(game_state)

        next_player = MultiAgentSearchAgent.SECOND_PLAYER if agent_index == MultiAgentSearchAgent.FIRST_PLAYER \
            else MultiAgentSearchAgent.FIRST_PLAYER
        evaluation = 0
        for action in actions:
            successor = game_state.generate_successor(agent_index, action)
            evaluation += self._max_player(successor, depth - 1, next_player, alpha, beta)

        return evaluation / float(len(actions))  # Uniform distribution


################################################################
# Heuristic
################################################################

# Configurations
WEIGHTS = {
    "smoothness": 0.1,
    "monotonicity": 1.0,
    "empty": 2.7,
    "max": 1.0
}

# Globals (should've been static variables, but we don't have a class :(... )
# transposition_table = {}  # TODO: Do we need to prune the transposition table sometimes? it get very big (1M+ entries)


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    # Setup
    board = current_game_state.board
    num_rows, num_cols = board.shape

    def in_board_bounds(row: int, col: int):
        """
        Return true if the given row and column are in the board bounds.
        :param row: The row to check.
        :param col: The column to check.
        :return: True if the given row and column are in the board bounds and false otherwise.
        """
        return 0 <= row <= num_rows - 1 and 0 <= col <= num_cols - 1

    def get_farhest_cell(row: int, col: int, direction: Tuple[int, int]):
        """
        Search for the first cell that is actually not occupied.

        Example:
        [[2 0 2 4]
         [0 2 8 4]
         [0 0 0 0]
         [0 0 0 2]]
         Yields for the direction vector (0, 1) -
         - point (0, 0) (value 2) => 2.
         - point (1, 1) (value 2) => 8.
         - point (1, 2) (value 8) => 4.
         - point (0, 3) (value 4) => 0. etc.

        :param row: The row to start from.
        :param col: The column to start from.
        :param direction: The direction to search with.
        :return: The cell value.
        """
        cell_value = 0
        while True:
            row += direction[0]
            col += direction[1]
            if not in_board_bounds(row, col):
                break

            cell_value = board[row, col]

            if cell_value != 0:
                break

        return cell_value

    def count_available_cells():
        """
        Counts the number of available cells.
        :return: The number of available cells.
        """
        return np.count_nonzero(board == 0)  # Empty cells marked with zero

    def evaluate_board_smoothness():
        """
        Evaluate the board smoothness.
        We sum the pairwise difference between neighboring tile (in log space, so it represents the number of actions
        required to happen before they can fuse).
        :return: The board smoothness value.
        """
        smoothness = 0

        for row in range(num_rows):
            for col in range(num_cols):
                # Is this an empty location?
                if board[row, col] == 0:
                    continue

                # Evaluate the value, by using a log to take into account the turns measurement.
                value = math.log2(board[row, col])

                # Measure the distances for right and bottom
                # (Note that we actually don't need to take up and left as it will be transposed)
                for direction in ((0, 1), (1, 0)):
                    farest_cell = get_farhest_cell(row, col, direction)

                    if farest_cell != 0:  # That's not an empty cell
                        smoothness -= abs(value - math.log2(farest_cell))

            return smoothness

    def evaluate_board_monotonicity():
        """
        Evaluate the board monotonicity. That is, evaluate if the cells are strictly increasing or decreasing.
        :return: The board monotonicity.
        """
        counters = [0, 0, 0, 0]

        for row in range(num_rows):
            current_index = 0
            next_index = current_index + 1
            while next_index < num_cols:
                # Find the closest cell that actually has a value
                while next_index < num_cols and board[row, next_index] == 0:
                    next_index += 1

                # Did we found something?
                if next_index >= num_rows:
                    next_index -= 1

                current_value = math.log2(board[row, current_index]) if board[row, current_index] != 0 else 0
                next_value = math.log2(board[row, next_index]) if board[row, next_index] != 0 else 0

                if current_value > next_value:
                    counters[0] += next_value - current_value
                else:
                    counters[1] += current_value - next_value

                current_index = next_index
                next_index += 1

        for col in range(num_cols):
            current_index = 0
            next_index = current_index + 1
            while next_index < num_rows:
                # Find the closest cell that actually has a value
                while next_index < num_rows and board[next_index, col] == 0:
                    next_index += 1

                # Did we found something?
                if next_index >= num_rows:
                    next_index -= 1

                current_value = math.log2(board[current_index, col]) if board[current_index, col] != 0 else 0
                next_value = math.log2(board[next_index, col]) if board[next_index, col] != 0 else 0

                if current_value > next_value:
                    counters[2] += next_value - current_value
                else:
                    counters[3] += current_value - next_value

                current_index = next_index
                next_index += 1

        return max(counters[0], counters[1]) + max(counters[2], counters[3])

    # board_key = str(board)  # We can't use numpy arrays as dictionary keys, but we can translate them into strings! :)
    # if board_key in transposition_table:
    #     transposition_table[board_key]['hits'] += 1
    #     return transposition_table[board_key]['score']

    # Prepare
    available_cells_count = count_available_cells()
    available_cells_count = math.log(available_cells_count) if available_cells_count > 0 else 0
    board_max_value = np.max(board)
    board_max_value = board_max_value * WEIGHTS['max'] if board_max_value > 0 else 0

    # Evaluate
    evaluation = evaluate_board_smoothness() * WEIGHTS['smoothness'] \
                 + evaluate_board_monotonicity() * WEIGHTS['monotonicity'] \
                 + available_cells_count * WEIGHTS['empty'] \
                 + board_max_value

    # Save in the transposition table
    # transposition_table[board_key] = {'hits': 0, 'score': evaluation}
    return evaluation


# Abbreviation
better = better_evaluation_function
