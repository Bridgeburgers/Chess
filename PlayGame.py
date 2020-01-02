import chess
import sys
import time
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

def PlayGame(player1, player2, pause=0.2, visual = 'svg'):
    """
    visual: "simple" | "svg" | None
    """
    use_svg = (visual == "svg")
    
    player1.color = True
    player2.color = False
    
    board = AIBoard()
    try:
        while not board.is_game_over(claim_draw=True):
            if board.turn == chess.WHITE:
                currentPlayer = player1
            else:
                currentPlayer = player2

            while True:
                uci = currentPlayer.Play(board)
                
                if IsLegal(uci, board):
                
                    board.push_uci(uci)
                    board_stop = DisplayBoard(board, use_svg)
        
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
    result = None
    if board.is_checkmate():
        msg = "checkmate: " + currentPlayer.Color() + " wins!"
        result = not board.turn
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
    return (result, msg, board)


    