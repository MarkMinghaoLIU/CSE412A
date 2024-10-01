# search.py
# ---------
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

class Node:
    def __init__(self, state, prev, action, priority=0): # state, previous node, action that brings to state 
        self.state = state
        self.prev = prev
        self.action = action
        self.priority = priority

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
    
    Start: (35, 1)
    Is the start a goal? False
    Start's successors: [((35, 2), 'North', 1), ((34, 1), 'West', 1)]
    """

    "*** YOUR CODE HERE ***"
    stack = util.Stack()  
    visited = set()       

    start_state = problem.getStartState()
    stack.push((start_state, []))  
    while not stack.isEmpty():
        current_state, actions = stack.pop()  

        if problem.isGoalState(current_state):
            return actions  

        if current_state not in visited:
            visited.add(current_state)  
            for successor, action, _ in problem.getSuccessors(current_state):
                stack.push((successor, actions + [action])) 
    return []  

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue  # Make sure to import the Queue
    queue = Queue()  # Initialize a queue for BFS
    visited = set()  # Set to track visited states
    start_state = problem.getStartState()
    queue.push((start_state, []))
    while not queue.isEmpty():
        current_state, actions = queue.pop()  

        if problem.isGoalState(current_state):
            return actions  

        if current_state not in visited:
            visited.add(current_state) 
            for successor, action, _ in problem.getSuccessors(current_state):
                queue.push((successor, actions + [action]))  

    return []  

    

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue  
    pq = PriorityQueue()  
    visited = set()  
    start_state = problem.getStartState()
    pq.push((start_state, [], 0), 0) 

    while not pq.isEmpty():
        current_state, actions, current_cost = pq.pop()  
        if problem.isGoalState(current_state):
            return actions  
        if current_state not in visited:
            visited.add(current_state)  
            for successor, action, step_cost in problem.getSuccessors(current_state):
                new_cost = current_cost + step_cost  
                pq.push((successor, actions + [action], new_cost), new_cost)  

    return []  


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """

    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    h = heuristic(problem.getStartState(), problem)
    frontier.push(Node(problem.getStartState(), None, None, h), h)
    expanded = set()
    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.state):
            actions = list()
            while node.action is not None:
                actions.append(node.action)
                node = node.prev
            actions.reverse()
            return actions
        
        if node.state not in expanded:
            expanded.add(node.state)
            for child in problem.getSuccessors(node.state):
                frontier.push(Node(child[0], node, child[1], child[2]+node.priority), child[2]+node.priority+heuristic(child[0], problem))
    return list() 



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
