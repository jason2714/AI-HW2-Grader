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
from game import Directions, Actions
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        # exit()
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
        newScore = successorGameState.getScore()
        foodAction = findNearestFoodAction(currentGameState)
        if foodAction == action:
            newScore += 0.05
        pacmanPos = successorGameState.getPacmanPosition()
        capsules = currentGameState.getCapsules()
        ghostsState = currentGameState.getGhostStates()
        for capsule in capsules:
            for ghostState in ghostsState:
                capsuleDistance = mazeDistance(successorGameState.getPacmanPosition(), capsule, successorGameState)
                if ghostState.scaredTimer == 0:
                    newScore += (1.0 / (capsuleDistance + 1e-1)) * 25
                elif capsuleDistance == 0:
                    newScore -= 350
                # else:
                #     newScore -= (1.0 / (mazeDistance(successorGameState.getPacmanPosition(), capsule,
                #                                      successorGameState) + 1e-1)) * 35
        for ghostState in ghostsState:
            # print 'current pacman position: {}'.format(currentGameState.getPacmanPosition())
            # print 'new pacman position: {}'.format(pacmanPos)
            # print 'ghost position: {}'.format(ghostState)
            ghostPos = tuple(map(int, ghostState.getPosition()))
            ghostDistance = mazeDistance(pacmanPos, ghostPos, successorGameState)
            if ghostState.scaredTimer <= 1:
                if ghostDistance == 1:
                    newScore -= 400
            else:
                pass
                # print 'scare!!!!!!!!!!!!!!'
                # print (1.0 / (ghostDistance + 1e-4)) * 1000

                if ghostDistance <= 1:
                    ghostStartPos = ghostState.start.getPosition()
                    ghostStartDistance = mazeDistance(pacmanPos, ghostStartPos, successorGameState)
                    if ghostStartDistance <= 1:
                        newScore -= 600
                newScore += (1.0 / (ghostDistance + 1)) * 300

        # print newScore
        # print
        return newScore  # default scoure
        # please change the return score as the score you want


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        """
        score, action = self.max_value(gameState, self.depth)
        return action

    def max_value(self, gameState, depth):
        legal_actions = gameState.getLegalActions(0)
        if depth == 0 or not legal_actions:
            return self.evaluationFunction(gameState), None
        max_score = float('-inf')
        best_action = None
        for action in legal_actions:
            # print action
            successorGameState = gameState.generateSuccessor(0, action)
            score = self.min_value(successorGameState, 1, depth)
            if score > max_score:
                max_score = score
                best_action = action
        return max_score, best_action

    def min_value(self, gameState, ghostIndex, depth):
        legal_actions = gameState.getLegalActions(ghostIndex)
        if not legal_actions:
            return self.evaluationFunction(gameState)
        min_score = float('inf')
        for action in legal_actions:
            successorGameState = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex == gameState.getNumAgents() - 1:
                score, _ = self.max_value(successorGameState, depth - 1)
            else:
                score = self.min_value(successorGameState, ghostIndex + 1, depth)
            # print 'score: {}, min_score: {}'.format(score, min_score)
            if score < min_score:
                min_score = score
        return min_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float('-inf')
        beta = float('inf')
        score, action = self.max_value(gameState, self.depth, alpha, beta)
        return action

    def max_value(self, gameState, depth, alpha, beta):
        legal_actions = gameState.getLegalActions(0)
        if depth == 0 or not legal_actions:
            return self.evaluationFunction(gameState), None
        best_action = None
        max_score = float('-inf')
        for action in legal_actions:
            successorGameState = gameState.generateSuccessor(0, action)
            score = self.min_value(successorGameState, 1, depth, alpha, beta)
            if score > max_score:
                max_score = score
                best_action = action
            alpha = max(max_score, alpha)
            if alpha > beta:
                break
        return max_score, best_action

    def min_value(self, gameState, ghostIndex, depth, alpha, beta):
        legal_actions = gameState.getLegalActions(ghostIndex)
        if not legal_actions:
            return self.evaluationFunction(gameState)
        min_score = float('inf')
        for action in legal_actions:
            successorGameState = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex == gameState.getNumAgents() - 1:
                score, _ = self.max_value(successorGameState, depth - 1, alpha, beta)
            else:
                score = self.min_value(successorGameState, ghostIndex + 1, depth, alpha, beta)
            min_score = min(score, min_score)
            beta = min(score, beta)
            if beta < alpha:
                break
        return min_score


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
        score, action = self.max_value(gameState, self.depth)
        return action

    def max_value(self, gameState, depth):
        legal_actions = gameState.getLegalActions(0)
        if depth == 0 or not legal_actions:
            return self.evaluationFunction(gameState), None
        max_score = float('-inf')
        best_action = None
        for action in legal_actions:
            # print action
            successorGameState = gameState.generateSuccessor(0, action)
            score = self.expect_value(successorGameState, 1, depth)
            if score > max_score:
                max_score = score
                best_action = action
        return max_score, best_action

    def expect_value(self, gameState, ghostIndex, depth):
        legal_actions = gameState.getLegalActions(ghostIndex)
        if not legal_actions:
            return self.evaluationFunction(gameState)
        total_score = 0
        for action in legal_actions:
            successorGameState = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex == gameState.getNumAgents() - 1:
                score, _ = self.max_value(successorGameState, depth - 1)
            else:
                score = self.expect_value(successorGameState, ghostIndex + 1, depth)
            total_score += score
        return total_score / len(legal_actions)


def general_serach(problem, structure, heuristic=lambda *x: 0):
    current_pos = problem.getStartState()
    visited = set()
    if isinstance(structure, util.PriorityQueue):
        structure.push((current_pos, [], 0), 0)
    else:
        structure.push((current_pos, [], 0))
    while not structure.isEmpty():
        pos, path, cost = structure.pop()
        if pos not in visited:
            visited.add(pos)
            if problem.isGoalState(pos):
                return path
            for next_state in problem.getSuccessors(pos):
                new_path = list(path)
                new_path.append(next_state[1])
                if isinstance(structure, util.PriorityQueue):
                    structure.push((next_state[0], new_path, cost + next_state[2]),
                                   cost + next_state[2] + heuristic(next_state[0], problem))
                else:
                    structure.push((next_state[0], new_path, next_state[2]))


def manhattanHeuristic(position, problem):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def aStarSearch(problem, heuristic=manhattanHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    return general_serach(problem, util.PriorityQueue(), heuristic)


def findNearestFoodAction(initGameState):
    visited = set()
    queue = util.PriorityQueue()
    queue.push((initGameState, [], 0), 0)
    while not queue.isEmpty():
        gameState, path, cost = queue.pop()
        pos = gameState.getPacmanPosition()
        if pos not in visited:
            visited.add(pos)
            next_x, next_y = pos
            if initGameState.getFood()[next_x][next_y]:
                return path[0]
            for action in gameState.getLegalPacmanActions():
                newGameState = gameState.generatePacmanSuccessor(action)
                new_path = list(path)
                new_path.append(action)
                queue.push((newGameState, new_path, cost + 1), cost + 1)


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = map(int, point2)
    x3, y3 = map(lambda x: int(round(x)), point2)
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    assert not walls[x3][y3], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    action_list = aStarSearch(prob)
    return len(action_list)


class PositionSearchProblem:
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost
