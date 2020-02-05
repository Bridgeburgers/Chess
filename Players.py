import chess
import random
import sys
import numpy as np
from scipy.stats import entropy

sys.path.append('D:/Documents/PythonCode/Chess')
import Heuristics
from MiscFunctions import RandomArgSort, Softmax, ProbDict, OpposingColor
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
    parentAction: the action used by the parent to retrieve this node
    populated: has this node been populated? (If not, it's an unvisited empty node)
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
        self.terminal = False
        self.parent = None
        self.parentAction = None
        self.root = False
        self.populated = False
        
    def PopulateNode(self, board, parent=None, rawPolicyDict=None, parentAction=None):
        """
        initializes a with all empty (unpopulated) nodes that are not yet "searched"
        the rawPolicyDict fed here must be evaluated on THIS node's board
        This is run separately from __init__ so that when empty nodes are created
        on this node, they don't all recursively fill their own nodes ad infinitum
        """
        self.populated = True
        self.board = board
        self.parent = parent
        self.parentAction = parentAction
        if self.parent is None:
            self.root = True
        if board.GetTerminalCondition():
            self.terminal = True
        actions = board.Actions()
        self.nodes = {action : MCTSNode(C=self.C, temp=self.temp) for action in actions}
        if rawPolicyDict is not None:
            self.rawP = rawPolicyDict
            self.P = ProbDict(rawPolicyDict, temp=self.temp)
            for a in actions:
                if self.nodes.get(a) is not None:
                    self.nodes[a].p = self.P[a]
            
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
    
    def UCBChildren(self):
        """
        get the UCB of all the node's children (even if unvisited)
        """
        ucb = {}
        for action in self.board.Actions():
            node = self.nodes[action]
            ucbVal = node.Q
            #if node.N > 0:
            ucbVal += node.C * np.sqrt(np.log(self.N) / (1+node.N))
            ucb[action] = ucbVal
        return ucb
        
    def UCBZeroChildren(self):
        ucb = {}
        for action in self.board.Actions():
            node = self.nodes[action]
            ucbVal = node.Q + node.C * self.P.get(action) * np.sqrt(self.N) / (1 + node.N)
            ucb[action] = ucbVal
        return ucb
        
class MCTS(HeuristicPlayer):
    """
    keep in mind that the output of policy is raw values, you must use
    Policy() to get the probability values
    """
    def __init__(self, N=300, maxSimulationDepth=20, C=2,
                 heuristic=Heuristics.RawNumericEvaluation,
                 policy = Heuristics.RawNumericPolicyEvaluation, ucbType='normal',
                 policyTemp=1, stochasticPlay=True, entropyIteration=False, color=False):
        super().__init__(heuristic, color)
        self.N = N #number of MCTS simulations per turn
        self.C = C #exploration constant
        self.maxSimulationDepth = maxSimulationDepth
        self.policy = policy
        self.temp = policyTemp
        self.stochasticPlay = stochasticPlay
        self.entropyIteration = entropyIteration
        self.ucbType = ucbType
    
    def Policy(self, board, actions):
        rawPolicyVals = self.policy(board, actions, color=board.Turn())
        probDict = ProbDict(rawPolicyVals, temp=self.temp)
        return probDict
    
    def SimulatePlay(self, board, color='W'):
        i = 0
        terminalStateReached = board.GetTerminalCondition() != False or i >= self.maxSimulationDepth
        while not terminalStateReached:
            actions = [move.uci() for move in board.legal_moves]
            p = self.Policy(board, actions)
            action = np.random.choice(actions, p=list(p.values()))
            board.PushUci(action)
            i += 1
            if board.GetTerminalCondition() or i >= self.maxSimulationDepth:
                terminalStateReached = True
        return self.heuristic(board, color)
    
    def VisitNode(self, node, action=None, parentNode=None, color='W', ucbType='normal'):
        """
        visit the given node, or create the root node if 'node' is None
        add 1 visit count to this node, and populate it if it's 'new'
        """
        node.N += 1
            
        if not node.populated:
            newNodeBoard = parentNode.board.MoveCopy(action)
            
            if ucbType.lower() == 'zero':
                rawPolicyDict = self.policy(
                    newNodeBoard, newNodeBoard.Actions(), color=newNodeBoard.Turn())
            else:
                rawPolicyDict = None
                
            node.PopulateNode(newNodeBoard, parent=parentNode,
                              rawPolicyDict=rawPolicyDict, parentAction=action)
            nodeVal = self.GetNodeValue(node, color=color)
            
            return -nodeVal
            
        if node.board.GetTerminalCondition():
            nodeVal = self.GetNodeValue(node, color=color)
            #node.Q = (node.N * node.Q + nodeVal) / (node.N + 1)
            node.Q = -nodeVal
            return -nodeVal
        
        #find the node with the highest UCB (random tie break) and call it with VisitNode
        if ucbType.lower() == 'zero':
            nodeUcb = node.UCBZeroChildren()
        else:
            nodeUcb = node.UCBChildren()
            
        bestActionInd = RandomArgSort(list(nodeUcb.values()), reverse=True)[0]
        bestAction = list(nodeUcb.keys())[bestActionInd]
        
        nodeToVisit = node.nodes.get(bestAction)
        nodeVal = self.VisitNode(nodeToVisit, action=bestAction,parentNode=node, 
                                 color=OpposingColor(color), ucbType=ucbType)
        
        node.Q = (node.N * node.Q - nodeVal) / (node.N + 1)
            
        if not node.root:
            return -nodeVal
        
    def Search(self, board, stochastic=True):
        rootNode = MCTSNode(C=self.C, temp=self.temp)
        rawPolicyDict = self.policy(board, board.Actions(), color=board.Turn())
        rootNode.PopulateNode(board, parent=None, rawPolicyDict=rawPolicyDict, parentAction=None)
        if self.entropyIteration:
            #probs = list(rootNode.P.values())
            #effectiveMoveNum = np.exp(entropy(probs))
            effectiveMoveNum = len(rootNode.nodes)
            nIterations = int(round(self.N * effectiveMoveNum))
        else:
            nIterations = self.N
        for _ in range(nIterations):
            self.VisitNode(rootNode, action=None, parentNode=None, color=self.Color(), 
                           ucbType=self.ucbType)
        #take the action with probability based on node visits
        actions = list(rootNode.nodes.keys())
        visits = [rootNode.nodes[a].N for a in actions]
        probs = [n / sum(visits) for n in visits]
        if stochastic:
            action = np.random.choice(actions, p=probs)
        else:
            actionInd = RandomArgSort(probs, reverse=True)[0]
            action = actions[actionInd]
        return action
    
    def Play(self, board):
        if board.turn != self.color:
            pass
        return self.Search(board, stochastic=self.stochasticPlay)
            
    def GetNodeValue(self, node, color='W'):
        """
        get the value of the given node by simulating play;
        for the zero version of this class, this will be overwritten by a nnet call
        """
        board = node.board.Copy()
        return self.SimulatePlay(board, color)
        
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