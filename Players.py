import chess
import random
import sys
import numpy as np

sys.path.append('D:/Documents/PythonCode/Chess')
import Heuristics
from MiscFunctions import RandomArgSort
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
        nextVals = [self.HashHeuristic(b, color) for b in nextBoards]
        boardSortInds = RandomArgSort(nextVals, reverse=True)
        return actions[boardSortInds[0]]
        

#%%
class ABPruner(HeuristicPlayer):
    
    def __init__(self, maxDepth=1, heuristic=Heuristics.RawNumericEvaluation,
                 color=False, alphaCeiling=1e8, verbose=False):
        super().__init__(heuristic, color)
        self.alphaCeiling = alphaCeiling
        self.maxDepth = maxDepth
        self.human = False
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
    
#%%
class MCTS(HeuristicPlayer):
    