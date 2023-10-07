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
import queue
from queue import PriorityQueue
import heapq
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
    
    
    
class PriorityQueue:
    def __init__(self):
        self.queue = []

    def put(self, item):
        self.queue.append(item)
        self.queue.sort(key=lambda x: x[0], reverse=True)

    def get(self):
        if not self.queue:
            return None

        item = self.queue.pop(0)
        return item

    def empty(self):
        return len(self.queue) == 0    
    
    

def depthFirstSearch(problem: SearchProblem):
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
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    fringe = util.Stack()
    visited = set()
    # Each item in the fringe is a tuple containing the state and a list of actions
    fringe.push((start_state, []))

    while not fringe.isEmpty():
        current_state, actions = fringe.pop()
        if problem.isGoalState(current_state):
            return actions
        visited.add(current_state)
        successors = problem.getSuccessors(current_state)
        for next_state, action, _ in successors:
                   if next_state not in visited:
                    next_actions = actions + [action]
                    fringe.push((next_state, next_actions))
    return []
    
def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start=problem.getStartState()
    fringe=util.Queue()
    visited=set()
    fringe.push((start,[]))
    while not fringe.isEmpty():
        current_state,action=fringe.pop()
        if problem.isGoalState(current_state):
            return action
        visited.add(current_state)
        successor=problem.getSuccessors(current_state)
        for next_state,actions,_ in successor:
                   if next_state not in visited:
                    next_actions = action+[actions]
                    fringe.push((next_state, next_actions))
    return []
   # util.raiseNotDefined()
   
def uniformCostSearch(problem:SearchProblem):
    start=problem.getStartState()
    node=PriorityQueue()
    path=[]
    visited=set([])  
    print("The starting point is ",start)
    node.put((0, (start, path)))
    dict={}
    while not node.empty():
        _,(current,path)=node.get()
        print("First node we get:",current)
        if problem.isGoalState(current):
           return path
        else:
         visited.add(current)
         print("The current_node we add:",current)
         successors=problem.getSuccessors(current)
         print("Successors functions works",successors)
         for state in successors:
            next_state,action,cost=state
            if next_state not in visited:
                newcost=cost
                if action == 'North' or action=='East':
                   newcost=cost+2
                elif action == 'South' or action=='West':
                   newcost=cost+1
                
                print("There is new cost",newcost)
                if next_state in dict:
                   newcost+=dict[next_state][1]
                new_path=path+ [action]
                node.put((newcost,(next_state,new_path)))
                
def uniformCostSearch1(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    nodes = PriorityQueue()
    path = []
    explored = set([])
    nodes.put((0, (problem.getStartState(), path)))
    dict = {} #used to store information about nodes, including their paths and costs.
    while not nodes.empty():
       _, (curr, path) = nodes.get() #this will dequeue the nodes with 
       if problem.isGoalState(curr):
        return path
       else:
        explored.add(curr)#set to mark it as explored.
        successors = problem.getSuccessors(curr)
        for state in successors:        
            next_node, action, cost = state
            if next_node not in explored:
              path_cost=cost
              if next_node in dict: #if nextnode exist in dict 
                 path_cost += dict[next_node][1]
              new_path = path + [action]# constructed by extending the current path with the action taken to reach the next_nod
              nodes.put((path_cost, (next_node, new_path)))
   # util.raiseNotDefined()
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0



def aStarSearch(problem: SearchProblem, heuristic=None):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    print("There is A* start",problem.getStartState())
    start_node=problem.getStartState()
    open_list=[]
    close_list=set([])
    path=[]
    heapq.heappush(open_list, (0, 0,start_node))
    while (open_list):
        _, current_g, current_node = heapq.heappop(open_list)
        if problem.isGoalState(current_node):
            return path
        close_list.add(current_node)
        successors = problem.getSuccessors(current_node)
        for successor, action, _ in successors:
            state_g = current_g + 1 
            state_h = heuristic(successor)
            state_f = state_g + state_h
            if successor not in close_list:
                state_in_open_list = any(node[1] == successor for node in open_list)
                    #print("Successor not in close list",state_in_open_list)
            if not state_in_open_list or state_g < current_g:
                heapq.heappush(open_list, (state_f, state_g, successor))
                         
    return []
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch