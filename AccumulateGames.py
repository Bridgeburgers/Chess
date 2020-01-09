import sys
import pandas as pd
sys.path.append('D:/Documents/PythonCode/Chess/')
from PlayGame import PlayGame
from Players import Player, ABPruner, FastPruner, MCTS

#%%

gameResults = []
for _ in range(20):
    l = PlayGame(ABPruner(maxDepth=4), ABPruner(maxDepth=4), 0)
    print(l)
    gameResults.append(l[0])