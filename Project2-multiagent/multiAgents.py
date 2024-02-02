# multiAgents.py
# --------------


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        food_lst = []
        for dot in newFood.asList():
            food_lst.append(util.manhattanDistance(dot, newPos))

        try:
            min_dist_to_food = min(food_lst)
        except ValueError:
            min_dist_to_food = 0

        score = successorGameState.getScore()
        score -= 0.5 * min_dist_to_food

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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
    def maxValue(self, s, depth):
        correct = None
        n = s[1].getNumAgents()
        v = float('-inf')
        for a in s[1].getLegalActions(0):
            nextAction = (a, s[1].generateSuccessor(0, a))
            if nextAction[1].isWin() or nextAction[1].isLose() or depth >= self.depth * n:
                temp = (a, self.evaluationFunction(nextAction[1]))
            else:
                temp = self.minValue1(nextAction, 1, depth + 1)
            if temp[1] > v:
                v = temp[1]
                correct = a
        return correct, v if correct is not None else (None, v)

    def minValue1(self, s, index, depth):
        v = float('inf')
        n = s[1].getNumAgents()
        correct = None
        for a in s[1].getLegalActions(index):
            nextAction = (a, s[1].generateSuccessor(index, a))
            if nextAction[1].isWin() or nextAction[1].isLose() or depth >= n * self.depth - 1:
                temp = (a, self.evaluationFunction(nextAction[1]))
            else:
                if s[1].getNumAgents() == 3:
                    temp = self.minValue2(nextAction, 2, depth + 1)
                else:
                    temp = self.maxValue(nextAction, depth + 1)
            if temp[1] < v:
                v = temp[1]
                correct = a
        return correct, v if correct is not None else (None, v)

    def minValue2(self, s, index, depth):
        v = float('inf')
        n = s[1].getNumAgents()
        correct = None
        for a in s[1].getLegalActions(index):
            nextAction = (a, s[1].generateSuccessor(index, a))
            if nextAction[1].isWin() or nextAction[1].isLose() or depth >= n * self.depth - 2:
                temp = (a, self.evaluationFunction(nextAction[1]))
            else:
                temp = self.maxValue(nextAction, depth + 1)
            if temp[1] < v:
                v = temp[1]
                correct = a
        return correct, v if correct is not None else (None, v)

    def getAction(self, gameState: GameState):
        # *** YOUR CODE HERE ***
        v = self.maxValue((0, gameState), 0)
        return v[0]
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def maxValue(self, s, depth, alpha, beta):
        correct = None
        n = s[1].getNumAgents()
        v = float('-inf')
        for a in s[1].getLegalActions(0):
            nextAction = (a, s[1].generateSuccessor(0, a))
            if nextAction[1].isWin() or nextAction[1].isLose() or depth >= self.depth * n:
                temp = (a, self.evaluationFunction(nextAction[1]))
            else:
                temp = self.minValue1(nextAction, 1, depth + 1, alpha, beta)
            if temp[1] > v:
                v = temp[1]
                correct = a

            if v > beta:
                return correct, v, alpha, beta if correct is not None else (None, v, alpha, beta)
            else:
                alpha = max(alpha, v)

        return correct, v, alpha, beta if correct is not None else (None, v, alpha, beta)

    def minValue1(self, s, index, depth, alpha, beta):
        v = float('inf')
        n = s[1].getNumAgents()
        correct = None
        for a in s[1].getLegalActions(index):
            nextAction = (a, s[1].generateSuccessor(index, a))
            if nextAction[1].isWin() or nextAction[1].isLose() or depth >= n * self.depth - 1:
                temp = (a, self.evaluationFunction(nextAction[1]))
            else:
                if s[1].getNumAgents() == 3:
                    temp = self.minValue2(nextAction, 2, depth + 1, alpha, beta)
                else:
                    temp = self.maxValue(nextAction, depth + 1, alpha, beta)
            if temp[1] < v:
                v = temp[1]
                correct = a

            if v < alpha:
                return correct, v, alpha, beta if correct is not None else (None, v, alpha, beta)
            else:
                beta = min(beta, v)

        return correct, v, alpha, beta if correct is not None else (None, v, alpha, beta)

    def minValue2(self, s, index, depth, alpha, beta):
        v = float('inf')
        n = s[1].getNumAgents()
        correct = None
        for a in s[1].getLegalActions(index):
            nextAction = (a, s[1].generateSuccessor(index, a))
            if nextAction[1].isWin() or nextAction[1].isLose() or depth >= n * self.depth - 2:
                temp = (a, self.evaluationFunction(nextAction[1]))
            else:
                temp = self.maxValue(nextAction, depth + 1, alpha, beta)
            if temp[1] < v:
                v = temp[1]
                correct = a

            if v < alpha:
                return correct, v, alpha, beta if correct is not None else (None, v, alpha, beta)
            else:
                beta = min(beta, v)

        return correct, v, alpha, beta if correct is not None else (None, v, alpha, beta)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        v = self.maxValue((0, gameState), 0, float('-inf'), float('inf'))
        return v[0]
        # util.raiseNotDefined()
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, game_state, depth, agent_index):
        """
        Expectimax algorithm to find the best move for the Pacman agent.

        Args:
            game_state (GameState): The current game state.
            depth (int): The current depth in the search tree.
            agent_index (int): The index of the current agent.

        Returns:
            tuple: A tuple containing the best value and action.
        """
        total_agents = game_state.getNumAgents()

        # Check for terminal states or reached maximum depth
        if game_state.isWin() or game_state.isLose() or depth == self.depth:
            return (self.evaluationFunction(game_state), None)

        if agent_index == 0:  # Maximizer (Pacman's turn)
            return self.maximize(game_state, depth, agent_index)

        else:  # Randomizer (Ghosts' turn)
            return self.randomize(game_state, depth, agent_index, total_agents)


    def maximize(self, game_state, depth, agent_index):
        """
        Helper function for the maximizer (Pacman).

        Args:
            game_state (GameState): The current game state.
            depth (int): The current depth in the search tree.
            agent_index (int): The index of the current agent.

        Returns:
            tuple: A tuple containing the best value and action for the maximizer.
        """
        legal_actions = game_state.getLegalActions(0)
        max_value = float('-inf')
        next_agent_index = 1

        for action in legal_actions:
            successor = game_state.generateSuccessor(0, action)
            evaluation, _ = self.expectimax(successor, depth, next_agent_index)

            if evaluation > max_value:
                max_value = evaluation
                best_move = action

        return (max_value, best_move)


    def randomize(self, game_state, depth, agent_index, total_agents):
        """
        Helper function for the randomizer (Ghosts).

        Args:
            game_state (GameState): The current game state.
            depth (int): The current depth in the search tree.
            agent_index (int): The index of the current agent.
            total_agents (int): The total number of agents in the game.

        Returns:
            tuple: A tuple containing the average value and a random action.
        """
        total_value = 0
        action_count = 0  # To calculate the number of total actions
        legal_actions = game_state.getLegalActions(agent_index)

        for action in legal_actions:
            action_count += 1
            successor = game_state.generateSuccessor(agent_index, action)

            if agent_index == total_agents - 1:
                evaluation, _ = self.expectimax(successor, depth + 1, 0)
                total_value += evaluation  # Calculate the total of all the values
                best_move = action
            else:
                evaluation, _ = self.expectimax(successor, depth, agent_index + 1)
                total_value += evaluation
                best_move = action

        average_value = total_value / action_count  # Calculate the average value

        return (average_value, best_move)


    def getAction(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction.

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        best = self.expectimax(game_state, 0, 0)
        return best[1]

    

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    totalscore, count, countf, fdist, fdistmax = 0, 0, 0, 0, 0
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghost = currentGameState.getGhostStates()
    ghostpos = currentGameState.getGhostPositions()
    newscared = [ghostState.scaredTimer for ghostState in ghost]
    score = currentGameState.getScore() + 0.51*max(newscared)   

    if currentGameState.isWin() ==  True:    
        score = 1000000
        return score

    elif currentGameState.isLose() == True:  
        score = -1000000
        return score

    for food in food.asList():
        fdist = manhattanDistance(food, pos)   
        if fdistmax == 0:
            fdistmax = fdist
        elif fdist < fdistmax:
            fdistmax = fdist

    if fdist < 3:          
        score += 10
    if fdistmax != 0:     
        score += 1/fdistmax 

    for ghost in ghostpos:
        gdist = manhattanDistance(pos, ghost)
        if gdist < 1:     
            score = -4839434
            return score
        elif gdist < 3:  
            score -= 100
            
    return score

    util.raiseNotDefined()



# Abbreviation
better = betterEvaluationFunction
