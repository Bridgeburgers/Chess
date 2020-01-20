from chess import Board
import numpy as np
#%%
class AIBoard(Board):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strState = str(self)
        
    def Copy(self):
        copyBoard = self.copy()
        copyBoard.strState = self.strState
        return copyBoard
    
    def BoardArray(self):
        a = np.array(list(self.strState))
        a = a[a!='\n']
        a = a[a!=' ']
        a = np.reshape(a, [8,8])
        return a

    def GetTerminalCondition(self):
        if self.is_checkmate():
            result = self.turn
            if result:
                return 'B'
            else:
                return 'W'
        elif self.is_stalemate():
            return 'D'
        elif self.is_fivefold_repetition():
            return 'D'
        elif self.is_insufficient_material():
            return 'D'
        elif self.can_claim_draw():
            return 'D'
        else:
            return False
        
    def PushUci(self, uci):
        self.push_uci(uci)
        self.strState = str(self)
        
    def TerminalConditionScore(self, color='W', scoreCeiling=1e7):
        t = self.GetTerminalCondition()
        if not t or t=='D':
            return 0
        elif t == color:
            return scoreCeiling
        else:
            return -scoreCeiling
     
    def Turn(self):
        if self.turn:
            return 'W'
        return 'B'
        
    def Actions(self):
        return [move.uci() for move in list(self.legal_moves)]
    
    def Move(self, action):
        self.PushUci(action)
        
    def MoveCopy(self, action):
        returnBoard = self.copy()
        returnBoard.Move(action)
        return returnBoard
    