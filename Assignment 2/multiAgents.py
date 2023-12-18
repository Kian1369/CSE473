# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        walls = successorGameState.getWalls()
        height, width = walls.height, walls.width
        pacmanPos = currentGameState.getPacmanPosition()
        #newFood = successorGameState.getFood()
        #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodGrid = currentGameState.getFood().asList()
        val = 0.0
        if action == 'Stop':
            val -= 1.1
        closestFood = (-1, -1)
        foodDistance = height * width

        for food in foodGrid:
            tempDist = manhattanDistance(pacmanPos, food)
            if tempDist < foodDistance or closestFood == (-1, -1):
                foodDistance = tempDist
                closestFood = food

        dist = manhattanDistance(newPos, closestFood)

        for ghost in newGhostStates:
            distanceOfGhost = manhattanDistance(newPos, ghost.getPosition())
            scaredTimes = ghost.scaredTimer
            if scaredTimes == 0:
                if distanceOfGhost <= 3:
                    val += -(4 - distanceOfGhost) ** 2
                else:
                    if currentGameState.hasFood(newPos[0], newPos[1]):
                        val += 1.5
            else:
                val += scaredTimes / (distanceOfGhost + 1)

        val = val - dist
        return val

def scoreEvaluationFunction(currentGameState):
        """
          This default evaluation function just returns the score of the state.
          The score is the same one displayed in the Pacman GUI.

          This evaluation function is meant for use with adversarial search agents
          (not reflex agents).
        """
        return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()
        total_depth = self.depth * num_agents

        # Remove 'STOP' from Pacman's action list if it exists.
        pacman_actions = gameState.getLegalActions(0)
        if Directions.STOP in pacman_actions:
            pacman_actions.remove(Directions.STOP)

        # Generate a list of Pacman's successor states.
        pacman_successors = []
        for action in pacman_actions:
            pacman_successors.append(gameState.generateSuccessor(0, action))

        # Calculate values for Pacman's successor states.
        successor_values = []
        for successor_state in pacman_successors:
            successor_values.append(self.MaxMinValue(successor_state, 1, num_agents, total_depth - 1))

        # Find the largest value(s) Pacman can obtain among all successor states.
        max_value = max(successor_values)
        best_indices = [index for index in range(len(successor_values)) if successor_values[index] == max_value]

        # Return the action that will maximize Pacman's value and randomly select one action if multiple have the greatest value.
        chosen_index = random.choice(best_indices)
        return pacman_actions[chosen_index]

    def MaxMinValue(self, gameState, agentIdx, numAgents, depth):
        """
        Calculate the greatest (smallest) value Pacman (ghost)
        can obtain among all successor states.
        """
        # Return the value of the current state using the evaluation function if the current state
        # is a terminal state (win/lose) or if the function reaches the specified depth.
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        legal_actions = gameState.getLegalActions(agentIdx)
        # Remove 'STOP' from the list of actions if the agent is Pacman.
        if agentIdx == 0:
            if Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)

        # Generate a list of successor states.
        successor_states = []
        for action in legal_actions:
            successor_states.append(gameState.generateSuccessor(agentIdx, action))

        # Evaluate the successor states by recursively calling this function until they reach a terminal state or the depth becomes 0.
        values = []
        for next_state in successor_states:
            values.append(self.MaxMinValue(next_state, (agentIdx + 1) % numAgents, numAgents, depth - 1))

        # If the agent is Pacman, return the maximum value, otherwise, return the minimum value.
        if agentIdx == 0:
            return max(values)
        else:
            return min(values)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Get legal moves for the current game state and the total number of agents in the game.
        legal_moves = gameState.getLegalActions()
        self.agent_num = gameState.getNumAgents()

        alpha = float("-inf")
        beta = float("inf")
        best_value = -float("inf")
        best_move = ""

        for move in legal_moves:
            # Generate a successor game state for Pacman and calculate the value for this move.
            successor = gameState.generateSuccessor(0, move)
            value = self.minValue(self.depth, successor, 1, alpha, beta)

            # Update the best move if a better one is found.
            if value > best_value:
                best_value = value
                best_move = move

            # Perform alpha-beta pruning to eliminate unnecessary branches.
            if value > beta:
                return best_move
            # Update alpha for the next iteration.
            alpha = max(alpha, best_value)
        return best_move

    def maxValue(self, current_depth, gameState, alpha, beta):
        """
        Calculate the maximum value for Pacman.
        """
        if current_depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        value = -float("inf")
        # Get legal moves for the current game state.
        legal_moves = gameState.getLegalActions()

        for move in legal_moves:
            successor = gameState.generateSuccessor(0, move)
            value = max(value, self.minValue(current_depth, successor, 1, alpha, beta))

            # Perform alpha-beta pruning to eliminate unnecessary branches.
            if value > beta:
                return value

            alpha = max(alpha, value)

        return value

    def minValue(self, current_depth, gameState, agent_index, alpha, beta):
        """
        Calculate the minimum value for the ghosts.
        """
        if current_depth == 0 or gameState.isLose() or gameState.isWin():
            # Return the evaluation function value for terminal states.
            return self.evaluationFunction(gameState)

        # Get legal moves for the current agent.
        legal_moves = gameState.getLegalActions(agent_index)
        value = float("inf")

        for move in legal_moves:
            successor = gameState.generateSuccessor(agent_index, move)

            if agent_index + 1 < self.agent_num:
                # Update the minimum value.
                value = min(value, self.minValue(current_depth, successor, agent_index + 1, alpha, beta))
            else:
                # Update the minimum value for Pacman's move.
                value = min(value, self.maxValue(current_depth - 1, successor, alpha, beta))

            # Perform alpha-beta pruning to eliminate unnecessary branches.
            if value < alpha:
                return value

            beta = min(beta, value)

        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Get legal moves for the current game state as well as the total number of agents in the game.
        legal_moves = gameState.getLegalActions()
        self.agent_num = gameState.getNumAgents()

        successors = []
        for move in legal_moves:
            # Generate successor game states for Pacman.
            successors.append(gameState.generateSuccessor(0, move))

        # Find the best score among all successor states.
        scores = [self.expectValue(self.depth, successor, 1) for successor in successors]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        # Randomly choose among the best moves.
        chosen_index = random.choice(best_indices)

        return legal_moves[chosen_index]

    def maxValue(self, current_depth, gameState):
        """
        Calculate the maximum value for Pacman.
        """
        if current_depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        # Get legal moves for the current game state.
        legal_moves = gameState.getLegalActions()
        successors = []
        for move in legal_moves:
            # Generate successor game states for Pacman.
            successors.append(gameState.generateSuccessor(0, move))

        # Calculate the maximum value among all successor states.
        values = [self.expectValue(current_depth, successor, 1) for successor in successors]
        return max(values)

    def expectValue(self, current_depth, gameState, agent_index):
        """
        Calculate the expected value for the ghosts.
        """
        # Return the evaluation function value for terminal states.
        if current_depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        # Get legal moves for the current agent.
        legal_moves = gameState.getLegalActions(agent_index)
        value = 0.0

        for move in legal_moves:
            # Generate successor game states for the agent.
            successor = gameState.generateSuccessor(agent_index, move)
            if agent_index + 1 < self.agent_num:
                value += self.expectValue(current_depth, successor, agent_index + 1)
            else:
                value += self.maxValue(current_depth - 1, successor)

        return value / len(legal_moves)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Define a base score for the current game state
    base_score = currentGameState.getScore()

    # Calculate the distance to the nearest food pellet
    food_distances = [manhattanDistance(pacmanPos, food) for food in foodGrid.asList()]
    if food_distances:
        closest_food_distance = min(food_distances)
    else:
        closest_food_distance = 0

    # Evaluate the impact of the closest ghost
    ghost_scores = []
    for i, ghostState in enumerate(ghostStates):
        ghost_pos = ghostState.getPosition()
        if scaredTimes[i] > 0:
            # If the ghost is scared, prioritize eating it
            ghost_scores.append(100 / (manhattanDistance(pacmanPos, ghost_pos) + 1))
        else:
            # If the ghost is not scared, avoid it
            ghost_scores.append(-100 / (manhattanDistance(pacmanPos, ghost_pos) + 1))

    # Calculate the total score based on various factors
    total_score = base_score - closest_food_distance + sum(ghost_scores)

    return total_score

# Abbreviation
better = betterEvaluationFunction