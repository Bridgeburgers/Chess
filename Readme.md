Chess AI Maker
==============

This is a module for making a chess AI and testing it by playing against it, or
pitting it against a standard AI created in this repo (e.g. alpha-beta using a
simple unit score heuristic).

To play against a chess AI in a Spyder editor, load the class from the file in
which the class was written, load PlayGame from *PlayGame.py*, load the *Player*
class from *Players.py*, and run PlayGame using both the *Player* class (with
*human* set to *True*), and the particular class as inputs. (First input is
white, second is black.) For example, if you want to play against the alpha-beta
pruner as black, open a Spyder editor and type the following:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from PlayGame import PlayGame
from Players import Player, ABPruner

PlayGame(ABPruner(maxDepth=3), Player(human=True))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the editor prompts you for a move, type the coordinates of the *from* cell
and the *to* cell. E.g.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
a2a4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To write an AI, create a child class of *Player*, and overwrite the method
*Play* with input *board*, that takes as input an instance of *AIBoard* (defined
in *AIBoard.py*) and outputs a string representing an action (e.g. *a2a4*).

 

I plan to add a RL-based AI for alpha-beta in the future.
