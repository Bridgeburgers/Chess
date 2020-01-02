import chess
import random
import sys
import numpy as np

sys.path.append('D:/Documents/PythonCode/Chess')
import Heuristics
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
        
    def Play(self, board):
        
        if board.turn != self.color:
            pass
        
        if self.human:
            move = input('Enter move: ')
            return move
        
        move = random.choice(list(board.legal_moves)).uci()
        return move

#%%
class ABPruner(Player):
    
    def __init__(self, human=False, color=False, 
                 heuristic=Heuristics.RawNumericEvaluation,
                 maxDepth=1, alphaCeiling=1e8):
        super().__init__(human, color)
        self.heuristic = heuristic
        self.alphaCeiling = alphaCeiling
        self.maxDepth = maxDepth
        
    def TerminalTest(self, board, depth, maxDepth):
        
        if depth >= maxDepth or board.GetTerminalCondition():
            return True
        return False
    
    def MaxValue(self, board, a, b, depth, maxDepth, color='W'):
        if self.TerminalTest(board, depth, maxDepth):
            return self.heuristic(board, color), ''
        
        v = -self.alphaCeiling
        actions = [move.uci() for move in list(board.legal_moves)]
        np.random.shuffle(actions)
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
            return self.heuristic(board, color), ''
        
        v = self.alphaCeiling
        actions = [move.uci() for move in list(board.legal_moves)]
        np.random.shuffle(actions)
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
        v, a = self.MaxValue(board, -self.alphaCeiling, self.alphaCeiling,
                             0, self.maxDepth, color)
        return a
    
    def Play(self, board):
        if board.turn != self.color:
            pass
        return self.ABSearch(board, self.Color())