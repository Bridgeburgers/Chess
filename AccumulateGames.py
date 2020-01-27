import sys
import pandas as pd
sys.path.append('D:/Documents/PythonCode/Chess/')
from PlayGame import PlayGame
from Players import Player, ABPruner, FastPruner, MCTS
import numpy as np

#%%
board = AIBoard()
PlayGame(ABPruner(2), MCTS(N=5000, maxSimulationDepth=0, policyTemp=40,
                           C=.1, stochasticPlay=False, ucbType='normal', entropyIteration=False), board=board)
#%%
PlayGame(ABPruner(3), MCTS(N=200, maxSimulationDepth=2, policyTemp=40, C=0.5, stochasticPlay=False, entropyIteration=True))
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