import chess
import numpy as np
import sys

sys.path.append('D:/Documents/PythonCode/Chess')
from MiscFunctions import Softmax, ProbDict
#%%
sMax = 4000
rawUnitScores = dict(
    P=100/sMax,
    N=320/sMax,
    B=330/sMax,
    R=500/sMax,
    Q=900/sMax,
    K=0
    )
keys = list(rawUnitScores.keys())
for k in keys:
    rawUnitScores[k.lower()] = -rawUnitScores[k]
    
rawUnitScores['.'] = 0
rawUnitScores['\n'] = 0
rawUnitScores[' '] = 0

#%%

def BasicNumericEvaluation(board, color='W', scoreCeiling=1,
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

def RawNumericEvaluation(board, color='W', scoreCeiling=1,
                         rawUnitScores=rawUnitScores):
    
    if board.GetTerminalCondition():
        return board.TerminalConditionScore(color, scoreCeiling)
    
    score = sum([rawUnitScores[c] for c in board.strState])
    if color.lower()=='b':
        score = -score
        
    return(score)

def RawNumericPolicyEvaluation(board, actions, color='W',
                               scoreCeiling=1, rawUnitScores=rawUnitScores):
    """
    non-normalized numeric policy evaluation
    """
    boards = [board.MoveCopy(a) for a in actions]
    scores = [RawNumericEvaluation(b, color, scoreCeiling, rawUnitScores)
              for b in boards]
    scoreDict = {a:s for a,s in zip(actions, scores)}
    return scoreDict

def NumericPolicyEvaluation(board, actions, color='W', temp=1,
                               scoreCeiling=1, rawUnitScores=rawUnitScores):
    scoreDict = RawNumericPolicyEvaluation(board, actions=actions, color=color,
                                           scoreCeiling=scoreCeiling,
                                           rawUnitScores=rawUnitScores)
    probDict = ProbDict(scoreDict)
    return probDict
    