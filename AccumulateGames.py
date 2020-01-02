import sys
import pandas as pd
sys.path.append('C:/Users/asibilia/Repos/Chess/')
from PlayGame import PlayGame
from Players import Player, ABPruner

#%%

gameResults = []
for _ in range(20):
    l = PlayGame(ABPruner(maxDepth=2), ABPruner(maxDepth=3), 0, visual=None)
    print(l)
    gameResults.append(l[0])