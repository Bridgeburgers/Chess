import chess
import random
import sys
import numpy as np

sys.path.append('D:/Documents/PythonCode/Chess')
import Heuristics
from MiscFunctions import RandomArgSort, Softmax, ProbDict
#%%

class Player:
    
    def __init__(self, human=False, color=False):
        self.human = human
        self.color = color
        
    def Color(self):
        if self.color:
            return "W"
        else:
            return "B"
        
    def SortCapturesFirst(self, moves, board):
        captureMoves = [move.uci() 
                        for move in list(board.generate_legal_captures())]
        nonCaptureMoves = [move for move in moves if not move in captureMoves]
        np.random.shuffle(nonCaptureMoves)
        np.random.shuffle(captureMoves)
        return captureMoves + nonCaptureMoves
    
    def Play(self, board):
        
        if board.turn != self.color:
            pass
        
        if self.human:
            move = input('Enter move: ')
            return move
        
        move = random.choice(list(board.legal_moves)).uci()
        return move

#%%
class HeuristicPlayer(Player):
    
    def __init__(self, heuristic=Heuristics.RawNumericEvaluation, color=False):
        super().__init__(False, color)
        self.heuristic = heuristic
        self.heuristicHash = {}
        
    def HashHeuristic(self, board, color):
        val = self.heuristicHash.get(board.strState)
        if val is None:
            val = self.heuristic(board, color)
            self.heuristicHash[board.strState] = val
        return val
    
    def Play(self, board):
        """
        just play the move with the highest heuristic value, random tiebreaker
        """
        actions = [move.uci() for move in list(board.legal_moves)]
        nextBoards = [board.MoveCopy(action) for action in actions]
        nextVals = [self.HashHeuristic(b, self.color) for b in nextBoards]
        boardSortInds = RandomArgSort(nextVals, reverse=True)
        return actions[boardSortInds[0]]
      
#%%
class MCTSNode:
    """
    board: the board state of this node
    nodes: dictionary of action-nodes 
        (contains all actions, including those with unexplored empty nodes)
    leaf: has no children (set to false when node is expanded)
    terminal: is end of a game
    root: has no parent
    Q: average utility (either from NN evaluation or SimulatePlay iterations)
    N: number of visits from MCTS iterations
    P: prior probability of forward node selections
    rawP: non-normalized prior probs; used to update P when new child nodes are created
    p: prior probability of node selection 
        (the element of parent's P that corresponds to itself)
    parent: reference to its parent node (None if root)
    """
    def __init__(self, C=2, temp=1):
        self.nodes = {}
        self.C = C
        self.temp = temp
        self.N = 0
        self.P = {}
        self.rawP = {}
        self.p = 1
        self.Q = 0
        self.leaf = True
        self.terminal = False
        self.parent = None
        self.root = False
        
    def PopulateNode(self, board, rawPolicyDict=None):
        """
        initializes a with all empty (unpopulated) nodes that are not yet "searched"
        This is run separately from __init__ so that when empty nodes are created
        on this node, they don't all recursively fill their own nodes ad infinitum
        """
        self.board = board
        actions = board.Actions()
        self.nodes = {action : MCTSNode(C=self.C, temp=self.temp) for action in actions}
        if rawPolicyDict is not None:
            self.rawP = rawPolicyDict
            self.P = ProbDict(rawPolicyDict)
            for a in actions:
                if self.nodes.get(a) is not None:
                    self.nodes[a].p = P[a]
            
        
    
    def CombineProbs(self, rawPolicyDict):
        """
        using dictionary of raw non-normalized policy values for new nodes and
        our own non-normalized policy values, update the normalized policy
        dictionary
        """
        self.rawP.update(rawPolicyDict)
        probDict = ProbDict(self.rawP)
        self.P = probDict
        for a in probDict.keys():
            if self.nodes.get(a) is not None:
                self.nodes[a].p = self.P[a]
    
    def UCB(self):
        if self.root:
            return 0
        
        ucb = self.Q + self.C * np.sqrt(np.log(self.parent.N) / self.N)
        return ucb
        
    def UCBZero(self):
        if self.root:
            return 0
        
        ucb = self.Q + self.C * self.p * np.sqrt(self.parent.N) / (1 + self.N)
        return ucb
    
class MCTS(HeuristicPlayer):
    """
    keep in mind that the output of policy is raw values, you must use
    Policy() to get the probability values
    """
    def __init__(self, N, maxSimulationDepth=100, C=2,
                 heuristic=Heuristics.RawNumericEvaluation,
                 policy = Heuristics.RawNumericPolicyEvaluation,
                 policyTemp=1, color=False):
        super().__init__(heuristic, color)
        self.N = N #number of MCTS simulations per turn
        self.C = C #exploration constant
        self.maxSimulationDepth = maxSimulationDepth
        self.policy = policy
        self.temp = policyTemp
    
    def Policy(self, board, actions):
        rawPolicyVals = self.policy(board, actions, color=self.color)
        probDict = ProbDict(rawPolicyVals)
        return probDict
    
    def SimulatePlay(self, board):
        i = 0
        terminalStateReached = False
        while not terminalStateReached:
            actions = [move.uci() for move in board.legal_moves]
            p = self.Policy(board, actions)
            action = np.random.choice(actions, p=list(p.values()))
            board.PushUci(action)
            
            if board.GetTerminalCondition() or i >= self.maxSimulationDepth:
                terminalStateReached = True
        return self.heuristic(board, self.color)
    
    def VisitNode(self, board, node=None):
        """
        if input node is None, this is the root iteration of this call
        """
        if node is None:
            node = MCTSNode(C=self.C, temp=self.temp)
        
#%%
class ABPruner(HeuristicPlayer):
    
    def __init__(self, maxDepth=1, heuristic=Heuristics.RawNumericEvaluation,
                 color=False, alphaCeiling=1e8, verbose=False):
        super().__init__(heuristic, color)
        self.alphaCeiling = alphaCeiling
        self.maxDepth = maxDepth
        self.verbose = verbose
        self.terminalStatesSearched = 0
    
    def TerminalTest(self, board, depth, maxDepth):
        
        if depth >= maxDepth or board.GetTerminalCondition():
            self.terminalStatesSearched += 1
            return True
        return False
    
    def MaxValue(self, board, a, b, depth, maxDepth, color='W'):
        if self.TerminalTest(board, depth, maxDepth):
            return self.HashHeuristic(board, color), ''
        
        v = -self.alphaCeiling
        actions = [move.uci() for move in list(board.legal_moves)]
        #np.random.shuffle(actions)
        actions = self.SortCapturesFirst(actions, board)
        
        maxAction = ''
        
        for action in actions:
            resBoard = board.MoveCopy(action)
            minRes, _ = self.MinValue(resBoard, a, b, depth+1, maxDepth, color)
            if minRes > v:
                maxAction = action
            v = max(v, minRes)
            if v >= b:
                return v, maxAction
            a = max(a, v)
        return v, maxAction
            
    def MinValue(self, board, a, b, depth, maxDepth, color='W'):
        if self.TerminalTest(board, depth, maxDepth):
            return self.HashHeuristic(board, color), ''
        
        v = self.alphaCeiling
        actions = [move.uci() for move in list(board.legal_moves)]
        #np.random.shuffle(actions)
        actions = self.SortCapturesFirst(actions, board)
        
        minAction = ''
        
        for action in actions:
            resBoard = board.MoveCopy(action)
            maxRes, _ = self.MaxValue(resBoard, a, b, depth+1, maxDepth, color)
            if maxRes < v:
                minAction = action
            v = min(v, maxRes)
            if v <= a:
                return v, minAction
            b = min(b, v)
        return v, minAction
    
    def ABSearch(self, board, color='W'):
        self.terminalStatesSearched = 0
        self.heuristicHash = {}
        v, a = self.MaxValue(board, -self.alphaCeiling, self.alphaCeiling,
                             0, self.maxDepth, color)
        if self.verbose:
            print(self.Color() + ': ' +\
                  str(self.terminalStatesSearched) + ' states searched')
        return a
    
    def Play(self, board):
        if board.turn != self.color:
            pass
        return self.ABSearch(board, self.Color())
    
#%%
class FastPruner(ABPruner):
    
    def MaxValue(self, board, a, b, depth, maxDepth, color='W', boardVal=0):
        if self.TerminalTest(board, depth, maxDepth):
            return boardVal, ''
        
        v = -self.alphaCeiling
        actions = [move.uci() for move in list(board.legal_moves)]
        
        #get the next boards resulting from possible actions and sort them by
        #their heuristic value
        nextBoards = [board.MoveCopy(action) for action in actions]
        nextVals = [self.HashHeuristic(b, color) for b in nextBoards]
        boardSortInds = RandomArgSort(nextVals, reverse=True)
        nextBoards = [nextBoards[i] for i in boardSortInds]
        nextVals = [nextVals[i] for i in boardSortInds]
        nextActions = [actions[i] for i in boardSortInds]
        
        maxAction = ''
        
        for nextBoard, val, action in zip(nextBoards, nextVals, nextActions):
            minRes, _ = self.MinValue(nextBoard, a, b, depth+1, 
                                      maxDepth, color, val)
            if minRes > v:
                maxAction = action
            v = max(v, minRes)
            if v >= b:
                return v, maxAction
            a = max(a, v)
        return v, maxAction
            
    def MinValue(self, board, a, b, depth, maxDepth, color='W', boardVal=0):
        if self.TerminalTest(board, depth, maxDepth):
            return boardVal, ''
        
        v = self.alphaCeiling
        actions = [move.uci() for move in list(board.legal_moves)]

        #get the next boards resulting from possible actions and sort them by
        #their heuristic value
        nextBoards = [board.MoveCopy(action) for action in actions]
        nextVals = [self.HashHeuristic(b, color) for b in nextBoards]
        boardSortInds = RandomArgSort(nextVals, reverse=True)
        nextBoards = [nextBoards[i] for i in boardSortInds]
        nextVals = [nextVals[i] for i in boardSortInds]
        nextActions = [actions[i] for i in boardSortInds]
        
        minAction = ''
        
        for nextBoard, val, action in zip(nextBoards, nextVals, nextActions):
            maxRes, _ = self.MaxValue(nextBoard, a, b, depth+1, 
                                      maxDepth, color, val)
            if maxRes < v:
                minAction = action
            v = min(v, maxRes)
            if v <= a:
                return v, minAction
            b = min(b, v)
        return v, minAction