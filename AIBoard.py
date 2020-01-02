from chess import Board
import numpy as np
#%%
class AIBoard(Board):
    
    def BoardArray(self):
        a = np.array(list(str(self)))
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
        
    def TerminalConditionScore(self, color='W', scoreCeiling=1e7):
        t = self.GetTerminalCondition()
        if not t or t=='D':
            return 0
        elif t == color:
            return scoreCeiling
        else:
            return -scoreCeiling
        
    def MoveCopy(self, move):
        returnBoard = self.copy()
        returnBoard.push_uci(move)
        return returnBoard