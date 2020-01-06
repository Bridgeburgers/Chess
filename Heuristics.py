import chess
import numpy as np
import sys

sys.path.append('D:/Documents/PythonCode/Chess')
from MiscFunctions import Softmax
#%%

rawUnitScores = dict(
    P=100,
    N=320,
    B=330,
    R=500,
    Q=900,
    K=0
    )
keys = list(rawUnitScores.keys())
for k in keys:
    rawUnitScores[k.lower()] = -rawUnitScores[k]
    
rawUnitScores['.'] = 0
rawUnitScores['\n'] = 0
rawUnitScores[' '] = 0

#%%

def BasicNumericEvaluation(board, color='W', scoreCeiling=1e6,
                           rawUnitScores=rawUnitScores):
    
    if board.GetTerminalCondition():
        return board.TerminalConditionScore(color, scoreCeiling)
    
    boardArray = board.BoardArray()
    score = 0
    for k in rawUnitScores.keys():
        score += rawUnitScores[k] * np.sum(boardArray==k)
        
    if color.lower()=='b':
        score = -score
        
    return score

def RawNumericEvaluation(board, color='W', scoreCeiling=1e6,
                         rawUnitScores=rawUnitScores):
    
    if board.GetTerminalCondition():
        return board.TerminalConditionScore(color, scoreCeiling)
    
    score = sum([rawUnitScores[c] for c in board.strState])
    if color.lower()=='b':
        score = -score
        
    return(score)

def RawNumericPolicyEvaluation(board, actions, color='W', temp=1,
                               scoreCeiling=1e6, rawUnitScores=rawUnitScores):
    boards = [board.MoveCopy(a) for a in actions]
    scores = [RawNumericEvaluation(b, color, scoreCeiling, rawUnitScores)
              for b in boards]
    probs = Softmax(scores, temp=temp)
    return probs
    