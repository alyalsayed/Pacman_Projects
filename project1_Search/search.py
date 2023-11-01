# search.py
# ---------

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    visited = set()  # Set to keep track of visited nodes
    frontier = util.Stack()  # Stack to store nodes for DFS (frontier)
    frontier.push((problem.getStartState(), []))  # Push start state with an empty path

    while not frontier.isEmpty():
        current_node, current_path = frontier.pop()  # Retrieve the current node and its path

        if problem.isGoalState(current_node):
            return current_path  # Return the path if the goal state is reached

        if current_node in visited:
            continue  # Skip if the current node has already been visited

        visited.add(current_node)  # Mark the current node as visited
        successors = problem.getSuccessors(current_node)  # Get successors of the current node

        for successor, action, cost in successors:
            new_path = current_path + [action]  # Append the action to the current path
            frontier.push((successor, new_path))  # Push the successor node with the updated path

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    visited = set()  # Set to keep track of visited nodes
    fringe = util.Queue()  # Queue to store nodes for BFS (fringe)
    start = problem.getStartState()  # Get the start point
    fringe.push((start, []))  # Push the start state into the empty queue

    while not fringe.isEmpty():
        current_node, current_path = fringe.pop()
        
        if problem.isGoalState(current_node):  # Check if we reached the goal state
            return current_path

        if current_node in visited:  # If the current state has already been visited, continue
            continue

        visited.add(current_node)  # Mark the current state as visited
        successors = problem.getSuccessors(current_node)

        for successor, action, cost in successors:
            new_path = current_path + [action]  # Add action to the current path
            fringe.push((successor, new_path))

    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    fringe = util.PriorityQueue()  # PriorityQueue to store nodes for UCS (fringe)
    cost_so_far = {}  # Dictionary to store the total cost to reach each state
    start_state = problem.getStartState()
    fringe.push((start_state, []), 0)  # Push the start state with an empty path and cost 0
    explored = set()  # Set to keep track of explored states

    while not fringe.isEmpty():
        current_state, current_path = fringe.pop()  # Retrieve the current state and its path
        
        if problem.isGoalState(current_state):
            return current_path  # Return the path if the goal state is reached

        if current_state in explored:
            continue  # Skip if the current state has already been explored

        explored.add(current_state)  # Mark the current state as explored
        successors = problem.getSuccessors(current_state)  # Get successors of the current state

        for successor, action, step_cost in successors:
            new_path = current_path + [action]  # Append the action to the current path
            new_cost = cost_so_far.get(current_state, 0) + step_cost  # Calculate the total cost to reach the successor

            if successor not in explored:
                cost_so_far[successor] = new_cost  # Update the cost to reach the successor
                fringe.push((successor, new_path), new_cost)  # Push the successor with the updated path and cost
    
    return -1  # Return -1 if no path is found

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    fringe = util.PriorityQueue()  # PriorityQueue to store nodes for A* search (fringe)
    cost_so_far = {}  # Dictionary to store the total cost to reach each state
    start_state = problem.getStartState()
    initial_cost = 0  # Initial cost is 0
    initial_heuristic = heuristic(start_state, problem)  # Calculate the heuristic for the start state
    fringe.push((start_state, []), initial_cost + initial_heuristic)  # Push the start state with an empty path and combined cost
    explored = set()  # Set to keep track of explored states

    while not fringe.isEmpty():
        current_state, current_path = fringe.pop()  # Retrieve the current state and its path
        
        if problem.isGoalState(current_state):
            return current_path  # Return the path if the goal state is reached

        if current_state in explored:
            continue  # Skip if the current state has already been explored

        explored.add(current_state)  # Mark the current state as explored
        successors = problem.getSuccessors(current_state)  # Get successors of the current state

        for successor, action, step_cost in successors:
            new_path = current_path + [action]  # Append the action to the current path
            new_cost = cost_so_far.get(current_state, 0) + step_cost  # Calculate the total cost to reach the successor

            if successor not in explored:
                cost_so_far[successor] = new_cost  # Update the cost to reach the successor
                heuristic_cost = heuristic(successor, problem)  # Calculate the heuristic cost for the successor
                total_cost = new_cost + heuristic_cost  # Calculate the combined cost of the successor
                fringe.push((successor, new_path), total_cost)  # Push the successor with the updated path and combined cost
    
    return -1  # Return -1 if no path is found

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch