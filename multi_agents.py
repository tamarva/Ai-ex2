import random

import numpy as np
import abc
import util
import math
from game import Agent, Action


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
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

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

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """

        def minmax(game_state, depth, agent_index):
            if depth == 0:
                return self.evaluation_function(game_state)

            actions = game_state.get_legal_actions(agent_index)
            if not actions:
                return self.evaluation_function(game_state)

            if agent_index == 0:
                value = -math.inf
                for action in actions:
                    successor = game_state.generate_successor(0, action)
                    val = minmax(successor, depth - 1, 1)
                    value = max(value, val)
                return value

            else:
                value = +math.inf
                for action in actions:
                    successor = game_state.generate_successor(1, action)
                    val = minmax(successor, depth - 1, 0)
                    value = min(value, val)
                return value

        legal_actions = game_state.get_legal_actions(0)
        successors = [game_state.generate_successor(0, action) for action in legal_actions]
        scores = [minmax(successor, (2 * self.depth) - 1, 1) for successor in successors]
        print(scores)
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        index_best_score = random.choice(best_indices)
        best_action = legal_actions[index_best_score]
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def alphabeta(game_state, depth, alpha, beta, agent_index):
            if depth == 0:
                return self.evaluation_function(game_state)

            actions = game_state.get_legal_actions(agent_index)
            if not actions:
                return self.evaluation_function(game_state)

            if agent_index == 0:
                value = -math.inf
                for action in actions:
                    successor = game_state.generate_successor(agent_index, action)
                    val = alphabeta(successor, depth - 1, alpha, beta, 1)
                    value = max(value, val)
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return value

            else:
                value = +math.inf
                for action in actions:
                    successor = game_state.generate_successor(agent_index, action)
                    val = alphabeta(successor, depth - 1, alpha, beta, 0)
                    value = min(value, val)
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
                return value

        legal_actions = game_state.get_legal_actions(0)
        successors = [game_state.generate_successor(0, action) for action in legal_actions]
        scores = [alphabeta(successor, (2 * self.depth) - 1, -math.inf, math.inf, 1) for successor in successors]
        best_score = max(scores)
        print(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        index_best_score = random.choice(best_indices)
        best_action = legal_actions[index_best_score]
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        util.raiseNotDefined()
        # same as minimax just to return in the min - the sum of the values of the children/ num of children



def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    """
    this evaluation prefers the states which have more free tiles - because if we have few free tiles with might end with low score
    """
    board = current_game_state.board
    upper_left_corner = board[0][0], upper_right_corner = board[0][-1], lower_right_corner = board[-1][-1], lower_left_corner = board[-1][0]
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
        highest_value_corner = find_max_val_corner()[0]
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
