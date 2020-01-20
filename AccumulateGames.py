import sys
import pandas as pd
sys.path.append('D:/Documents/PythonCode/Chess/')
from PlayGame import PlayGame
from Players import Player, ABPruner, FastPruner, MCTS

#%%
PlayGame(ABPruner(4), MCTS(N=400, maxSimulationDepth=0, policyTemp=40,
                           C=0.03, stochasticPlay=False, ucbType='normal', entropyIteration=True))

PlayGame(ABPruner(3), MCTS(N=200, maxSimulationDepth=2, policyTemp=40, C=0.5, stochasticPlay=False, entropyIteration=True))
#%%
gameResults = []
for _ in range(20):
    l = PlayGame(ABPruner(maxDepth=4), ABPruner(maxDepth=4), 0)
    print(l)
    gameResults.append(l[0])