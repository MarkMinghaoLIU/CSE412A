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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"
        # Finding the Nearest Food
        Foods = newFood.asList()
        d_nearest_food = float('inf')
        if not Foods: 
            d_nearest_food = 0
        else:
            for food in Foods:
                d_nearest_food = min(d_nearest_food, manhattanDistance(food, newPos))


        # Finding the Nearest Active Ghost (the ghost is not scared)/
        d_nearest_ghost = float('inf')
        for ghost_state in newGhostStates:
            g_x, g_y = ghost_state.getPosition()
            if ghost_state.scaredTimer == 0:
                d_nearest_ghost = min(d_nearest_ghost, manhattanDistance((g_x, g_y), newPos))
        d_nearest_ghost +=1 # avoid being 0
        return successorGameState.getScore() - d_nearest_food/5 -4/d_nearest_ghost

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
        _, bestAction = self.minimax(0, 0, gameState)
        return bestAction

    def minimax(self, agentIndex, depth, gameState):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        if agentIndex == 0:  
            return self.max_value(agentIndex, depth, gameState)
        else:  
            return self.min_value(agentIndex, depth, gameState)

    def max_value(self, agentIndex, depth, gameState):
        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState), Directions.STOP
        
        bestScore = float('-inf')
        bestAction = Directions.STOP
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            score, _ = self.minimax(1, depth, successor)  
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestScore, bestAction

    def min_value(self, agentIndex, depth, gameState):
        actions = gameState.getLegalActions(agentIndex)
        if not actions:  
            return self.evaluationFunction(gameState), Directions.STOP
        
        bestScore = float('inf')
        bestAction = Directions.STOP
        numAgents = gameState.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        if nextAgent == 0:
            nextDepth = depth + 1
        else:
            nextDepth = depth

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            score, _ = self.minimax(nextAgent, nextDepth, successor)
            if score < bestScore:
                bestScore = score
                bestAction = action

        return bestScore, bestAction



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the alpha-beta action using self.depth and self.evaluationFunction
        """
        _, bestAction = self.alphabeta(gameState, agentIndex=0, depth=self.depth, alpha=float('-inf'), beta=float('inf'))
        return bestAction
    
    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        
        if agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.min_value(gameState, agentIndex, depth, alpha, beta)
        
    def max_value(self, gameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState), Directions.STOP
        
        bestScore = float('-inf')
        bestAction = Directions.STOP
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            score, _ = self.alphabeta(successor, 1, depth, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action

            alpha = max(alpha, bestScore)
            if alpha > beta:  
                break

        return bestScore, bestAction

    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState), Directions.STOP

        bestScore = float('inf')
        bestAction = Directions.STOP
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            score, _ = self.alphabeta(successor, nextAgent, nextDepth, alpha, beta)
            if bestScore > score:
                bestScore = score
                bestAction = action
            beta = min(beta, bestScore)
            if alpha > beta: 
                break

        return bestScore, bestAction



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
        _, action = self.ExpectedmaxSearch(gameState, agentIndex=0, depth=self.depth)
        return action
    
    def ExpectedmaxSearch(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth==0:
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.maximizer(gameState, agentIndex, depth)
        else:
            return self.expectation(gameState, agentIndex, depth)
        
    def maximizer(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        maxScore = float('-inf')
        maxAction = Directions.STOP
        next_agent, next_depth = (0, depth - 1) if agentIndex == gameState.getNumAgents() - 1 else (agentIndex + 1, depth)

        for action in actions:
            successor_state = gameState.generateSuccessor(agentIndex, action)
            new_score, _ = self.ExpectedmaxSearch(successor_state, next_agent, next_depth)
            if new_score > maxScore:
                maxScore, maxAction = new_score, action
        return maxScore, maxAction

    def expectation(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        expScore = 0
        next_agent, next_depth = (0, depth - 1) if agentIndex == gameState.getNumAgents() - 1 else (agentIndex + 1, depth)

        for action in actions:
            successor_state = gameState.generateSuccessor(agentIndex, action)
            new_score, _ = self.ExpectedmaxSearch(successor_state, next_agent, next_depth)
            expScore += new_score
        expScore /= len(actions)
        return expScore, Directions.STOP
    

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    The insight is to get closed to the nearest food while keeping away from the ghosts that are active.
    Since when minimum distance to the active ghosts is large, the goal of pacman is to find the nearest food, 
    so I put the factor of distance to ghost on the denominator so that when it is large, the distance to the nearest food is the main factor.
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Foods = currentGameState.getFood().asList()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    # Finding the Nearest Food
    d_nearest_food = float('inf')
    if not Foods: 
        d_nearest_food = 0
    else:
        for food in Foods:
            d_nearest_food = min(d_nearest_food, manhattanDistance(food, Pos))


    # Finding the Nearest Active Ghost (the ghost is not scared)
    d_nearest_ghost = float('inf')
    for ghost_state in GhostStates:
        g_x, g_y = ghost_state.getPosition()
        if ghost_state.scaredTimer == 0:
            d_nearest_ghost = min(d_nearest_ghost, manhattanDistance((g_x, g_y), Pos))
    d_nearest_ghost +=1 # avoid being 0
    return currentGameState.getScore() - d_nearest_food/5 -4/d_nearest_ghost


# Abbreviation
better = betterEvaluationFunction