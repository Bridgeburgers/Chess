import chess
import sys
import time
import timeit
import numpy as np
from IPython.display import display, HTML, clear_output

sys.path.append('D:/Documents/PythonCode/Chess')

import Players
from AIBoard import AIBoard

#%%
def DisplayBoard(board, use_svg):
    if use_svg:
        return board._repr_svg_()
    else:
        return "<pre>" + str(board) + "</pre>"
    
def IsLegal(move, board):
    return chess.Move.from_uci(move) in board.legal_moves
#%%

def PlayGame(player1, player2, pause=0.2, visual = 'svg', 
             board=None, displayDuration=False, saveStates=False):
    """
    visual: "simple" | "svg" | None
    """    
    use_svg = (visual == "svg")
    
    player1.color = True
    player2.color = False
    
    if board is None:
        board = AIBoard()
        
    states = []
    
    try:
        while not board.is_game_over(claim_draw=True):
            if board.turn == chess.WHITE:
                currentPlayer = player1
                currentColor = 'W'
            else:
                currentPlayer = player2
                currentColor = 'B'

            while True:
                if displayDuration: startTime = timeit.default_timer()
                uci = currentPlayer.Play(board)
                if displayDuration:
                    elapsed = timeit.default_timer() - startTime
                    print(currentColor + ': ' + str(elapsed) + ' seconds')
                
                if IsLegal(uci, board):
                    board.Move(uci, storePrevious=True)
                    board_stop = DisplayBoard(board, use_svg)
                    
                    if saveStates:
                        states.append(board.BoardCNNState())
        
                    if visual is not None:
                        if visual == "svg":
                            clear_output(wait=True)
        
                        display(board)
                        if visual == "svg" and not currentPlayer.human:
                            time.sleep(pause)
                    break
                
    except KeyboardInterrupt:
        msg = "Game interrupted!"
        return (None, msg, board)
    
    states = np.array(states)
    
    result = 0
    if board.is_checkmate():
        msg = "checkmate: " + currentPlayer.Color() + " wins!"
        result = -1 + 2 * (not board.turn)
    elif board.is_stalemate():
        msg = "draw: stalemate"
    elif board.is_fivefold_repetition():
        msg = "draw: 5-fold repetition"
    elif board.is_insufficient_material():
        msg = "draw: insufficient material"
    elif board.can_claim_draw():
        msg = "draw: claim"
    if visual is not None:
        print(msg)
        
    return (result, msg, board, states)


    