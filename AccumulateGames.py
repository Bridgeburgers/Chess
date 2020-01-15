import sys
import pandas as pd
sys.path.append('D:/Documents/PythonCode/Chess/')
from PlayGame import PlayGame
from Players import Player, ABPruner, FastPruner, MCTS

#%%
PlayGame(ABPruner(3), MCTS(N=300, maxSimulationDepth=5, policyTemp=40, C=0.1, stochasticPlay=False))

PlayGame(MCTS(N=300, maxSimulationDepth=5, policyTemp=40, C=0.1, stochasticPlay=False), ABPruner(3))
#%%
gameResults = []
for _ in range(20):
    l = PlayGame(ABPruner(maxDepth=4), ABPruner(maxDepth=4), 0)
    print(l)
    gameResults.append(l[0])