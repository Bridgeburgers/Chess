import sys
sys.path.append('D:/Documents/PythonCode/Chess/')
from PlayGame import PlayGame
from Players import Player, ABPruner, FastPruner, MCTS
from AIBoard import AIBoard
from ValueNetwork import ValueNetwork
import tensorflow as tf
import numpy as np

#%%
#create full game data sets
nGames = 2
gameSets = []
gameResults = []
msgs = []
boards = []

for _ in range(nGames):
    res, msg, finalBoard, states = PlayGame(ABPruner(2), ABPruner(2), pause=0, visual=None, saveStates=True)
    gameSets.append(states)
    gameResults.append(float(res))
    msgs.append(msg)
    boards.append(finalBoard)
    
#%%
model = ValueNetwork(3, 12, 3, 64, 0.5)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mseLoss = tf.keras.losses.MeanSquaredError()

model.compile(optimizer, loss=mseLoss)

#%%
i = 1
xTrain = gameSets[i]
nMoves = xTrain.shape[0]
yTrain = np.repeat(gameResults[i], nMoves)

model.fit(xTrain, yTrain, epochs=1, batch_size=nMoves)
