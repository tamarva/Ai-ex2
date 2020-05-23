import random
from enum import Enum, auto
from typing import Callable

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

        # TODO Add edge cases handlers (depth < 2)

        legal_actions = game_state.get_legal_actions(0)
        successors = [game_state.generate_successor(0, action) for action in legal_actions]
        scores = np.array([self._max_player(successor, 1, MultiAgentSearchAgent.SECOND_PLAYER)
                           for successor in successors])
        print(f'Scores: {scores} :: Selected Move: {legal_actions[scores.argmax(axis=0)]} (idx={scores.argmax(axis=0)})')
        return legal_actions[scores.argmax(axis=0)]

    def _evaluate(self, game_state: GameState, depth: int, agent_index: int, alpha=-np.inf, beta=+np.inf):
        # If we're at a terminal node, evaluate it by using the heuristic value
        if self.depth == depth:
            return self.evaluation_function(game_state)

        # If we're the first player, we should use the max-player method
        if agent_index == MultiAgentSearchAgent.FIRST_PLAYER:
            return self._max_player(game_state, depth + 1, agent_index, alpha, beta)
        else:
            return self._min_player(game_state, depth + 1, agent_index, alpha, beta)

    def _max_player(self, game_state: GameState, depth: int, agent_index: int, alpha=-np.inf, beta=+np.inf):
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
        actions = game_state.get_legal_actions(agent_index)
        if not actions:
            return self.evaluation_function(game_state)

        max_score = -np.inf
        for action in actions:
            successor = game_state.generate_successor(agent_index, action)
            max_score = max(max_score, self._evaluate(successor, depth,  MultiAgentSearchAgent.SECOND_PLAYER,
                                                      alpha, beta))

            # Alpha-Beta cutoff
            if self.algorithm == SearchAlgorithm.ALPHABETA:
                if max_score > beta:
                    break
                alpha = max(alpha, max_score)

        return max_score

    def _min_player(self, game_state: GameState, depth: int, agent_index: int, alpha, beta):
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
        actions = game_state.get_legal_actions(agent_index)
        if not actions:
            return self.evaluation_function(game_state)

        min_score = np.inf
        for action in actions:
            successor = game_state.generate_successor(agent_index, action)
            min_score = min(min_score, self._evaluate(successor, depth,  MultiAgentSearchAgent.FIRST_PLAYER,
                                                      alpha, beta))

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

    def _min_player(self, game_state: GameState, depth: int, agent_index: int, alpha=-np.inf, beta=+np.inf):
        """
        Perform the minimizing player move in the expectimax algorithm.
        :param game_state: The current game state.
        :param depth: The depth of the search.
        :param agent_index: The player agent index.
        :param alpha: The alpha-value.
        :param beta: The beta-value.
        :return: The move score or best move.
        """
        # Iterate over the available actions
        actions = game_state.get_legal_actions(agent_index)
        if not actions:
            return self.evaluation_function(game_state)

        next_player = MultiAgentSearchAgent.SECOND_PLAYER if agent_index == MultiAgentSearchAgent.FIRST_PLAYER \
            else MultiAgentSearchAgent.FIRST_PLAYER

        evaluation = 0
        for action in actions:
            successor = game_state.generate_successor(agent_index, action)
            evaluation += self._evaluate(successor, depth, next_player, alpha, beta)

        return evaluation / float(len(actions))  # Uniform distribution


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    """
    this evaluation prefers the states which have more free tiles - because if we have few free tiles with might end with low score
    """
    board = current_game_state.board
    upper_left_corner, upper_right_corner, lower_right_corner, lower_left_corner = \
        board[0, 0], board[0, -1], board[-1, -1], board[-1, 0]
    corners = [upper_right_corner, upper_left_corner, lower_right_corner, lower_left_corner]
    num_rows, num_cols = board.shape

    def smoothness_eval():
        """
        this method tries to ensure we prefer states where its easier to fuse tiles and get higher scores
        """
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

    def more_free_tiles_evaluation():
        empty_tiles = current_game_state.get_empty_tiles()
        return empty_tiles[0].size

    def monotonic_snake_eval():
        """
      this heuristic goal is to ensure that we don't have a large value between small values- which will make it more difficult to merge the
      tiles and get higher score. so we want to keep the values in monotonic way.
      for each corner there are 2 possible "snakes" so i find the highest value on board and from there i calculate the 2 snakes sum and return
      the maximum
      """
        highest_value_corner = find_max_val_corner()
        if highest_value_corner == upper_right_corner:
            my_score = calculate_score_upper_right()

        if highest_value_corner == upper_left_corner:
            my_score = calculate_score_upper_left()

        if highest_value_corner == lower_left_corner:
            my_score = calculate_score_lower_left()

        if highest_value_corner == lower_right_corner:
            my_score = calculate_score_lower_right()

        return my_score

    def calculate_score_lower_right():
        def calculate_left_direc():
            some_val = 0.25
            weight = 1
            sum_score = 0
            to_turn_over = True
            for row_index in range(num_rows - 1, -1, -1):
                for col_index in range(num_cols):
                    row_idx = row_index
                    col_idx = col_index
                    if to_turn_over:
                        col_idx = num_cols - col_idx - 1
                    sum_score += board[row_idx][col_idx] * weight
                    weight = weight * some_val
                to_turn_over = not to_turn_over
            return sum_score

        def calculate_up_direc():
            some_val = 0.25
            weight = 1
            sum_score = 0
            to_turn_over = True
            for col_index in range(num_cols - 1, -1, -1):
                for row_index in range(num_rows):
                    row_idx = row_index
                    col_idx = col_index
                    if to_turn_over:
                        row_idx = num_rows - row_idx - 1
                    sum_score += board[row_idx][col_idx] * weight
                    weight = weight * some_val
                to_turn_over = not to_turn_over
            return sum_score

        return max(calculate_up_direc(), calculate_left_direc())

    def calculate_score_upper_right():
        def calculate_left_direc():
            some_val = 0.25
            weight = 1
            sum_score = 0
            to_turn_over = False
            for row_index in range(num_rows):
                for col_index in range(num_cols):
                    row_idx = row_index
                    col_idx = col_index
                    if to_turn_over:
                        col_idx = num_cols - col_idx - 1
                    sum_score += board[row_idx][col_idx] * weight
                    weight = weight * some_val
                to_turn_over = not to_turn_over
            return sum_score

        def calculate_down_direc():
            some_val = 0.25
            weight = 1
            sum_score = 0
            to_turn_over = False
            for col_index in range(num_cols - 1, -1, -1):
                for row_index in range(num_rows):
                    row_idx = row_index
                    col_idx = col_index
                    if to_turn_over:
                        row_idx = num_rows - row_idx - 1
                    sum_score += board[row_idx][col_idx] * weight
                    weight = weight * some_val
                to_turn_over = not to_turn_over
            return sum_score

        return max(calculate_down_direc(), calculate_left_direc())

    def calculate_score_upper_left():
        def calculate_right_direc():
            some_val = 0.25
            weight = 1
            sum_score = 0
            to_turn_over = False
            for row_index in range(num_rows):
                for col_index in range(num_cols):
                    row_idx = row_index
                    col_idx = col_index
                    if to_turn_over:
                        col_idx = num_cols - col_idx - 1
                    sum_score += board[row_idx][col_idx] * weight
                    weight = weight * some_val
                to_turn_over = not to_turn_over
            return sum_score

        def calculate_down_direc():
            some_val = 0.25
            weight = 1
            sum_score = 0
            to_turn_over = False
            for col_index in range(num_rows):
                for row_index in range(num_cols):
                    row_idx = row_index
                    col_idx = col_index
                    if to_turn_over:
                        row_idx = num_rows - row_idx - 1
                    sum_score += board[row_idx][col_idx] * weight
                    weight = weight * some_val
                to_turn_over = not to_turn_over
            return sum_score

        return max(calculate_down_direc(), calculate_right_direc())

    def calculate_score_lower_left():
        def calculate_right_direc():
            some_val = 0.25
            weight = 1
            sum_score = 0
            to_turn_over = False
            for row_index in range(num_rows - 1, -1, -1):
                for col_index in range(num_cols):
                    row_idx = row_index
                    col_idx = col_index
                    if to_turn_over:
                        col_idx = num_cols - col_idx - 1
                    sum_score += board[row_idx][col_idx] * weight
                    weight = weight * some_val
                to_turn_over = not to_turn_over
            return sum_score

        def calculate_up_direc():
            some_val = 0.25
            weight = 1
            sum_score = 0
            to_turn_over = True
            for col_index in range(num_cols):
                for row_index in range(num_rows):
                    row_idx = row_index
                    col_idx = col_index
                    if to_turn_over:
                        row_idx = num_rows - row_idx - 1
                    sum_score += board[row_idx][col_idx] * weight
                    weight = weight * some_val
                to_turn_over = not to_turn_over
            return sum_score

        return max(calculate_up_direc(), calculate_right_direc())

    def find_max_val_corner():
        return max([corner for corner in corners])

    return 0.6 * monotonic_snake_eval() + 0.2 * more_free_tiles_evaluation() + 0.2 * smoothness_eval()


# Abbreviation
better = better_evaluation_function
