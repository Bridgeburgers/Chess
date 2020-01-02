import chess
import numpy as np
#%%

rawUnitScores = dict(
    P=100,
    N=320,
    B=330,
    R=500,
    Q=900
    )
keys = list(rawUnitScores.keys())
for k in keys:
    rawUnitScores[k.lower()] = -rawUnitScores[k]

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