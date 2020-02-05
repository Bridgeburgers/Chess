import sys
import pandas as pd
sys.path.append('D:/Documents/PythonCode/Chess/')
from PlayGame import PlayGame
from Players import Player, ABPruner, FastPruner, MCTS
from AIBoard import AIBoard
import numpy as np

#%%
board = AIBoard()
PlayGame(ABPruner(3), MCTS(N=8000, maxSimulationDepth=0, policyTemp=40,
                           C=.3, stochasticPlay=False, ucbType='normal', entropyIteration=False), board=board)
#%%
PlayGame(ABPruner(2), ABPruner(3), board=board)
#%%
gameResults = []
for _ in range(20):
    l = PlayGame(ABPruner(maxDepth=4), ABPruner(maxDepth=4), 0)
    print(l)
    gameResults.append(l[0])
    
#%%
exploreConstants = np.arange(0.05, 1.55, step=0.05)
resDict = {}
for c in exploreConstants:
    gameRes = PlayGame(ABPruner(2), 
                       MCTS(N=1000, maxSimulationDepth=0, C=c,
                            stochasticPlay=False, ucbType='normal', entropyIteration=False),
                       visual=None)
    resDict[c] = gameRes