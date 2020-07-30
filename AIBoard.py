from chess import Board
import numpy as np
#%%
class AIBoard(Board):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strState = str(self)
        self.history = []
        
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
    
    def Move(self, action, storePrevious=False):
        if storePrevious:
            self.history.append(self.Copy())
        self.PushUci(action)
        
    def MoveCopy(self, action):
        returnBoard = self.copy()
        returnBoard.Move(action)
        return returnBoard
    
    def UndoMove(self, inplace=True):
        if len(self.history) > 0:
            history = self.history
            self.__dict__.update(history[len(history)-1].__dict__)
            del history[len(history)-1]
            self.history = history
            
    def BoardCNNState(self):
        """
        get a numpy array of the board in a format for CNN training
        """
        x = self.BoardArray()
        
        #get arrays for pieces
        pieces = ['P','R','N','B','Q','K','p','r','n','b','q','k']
        X = np.stack([(x==s).astype(float) for s in pieces], axis=2)
        return X
        
        zeros = np.zeros([8,8,1], dtype=float)
        
        #get castling arrays
        p1Castle = zeros + self.has_castling_rights(0)
        p2Castle = zeros + self.has_castling_rights(1)
        
        turn = zeros + self.turn
        
        repetitions = 1
        for i in range(2,6):
            if self.is_repetition(i):
                repetitions = i
            else:
                break
        repetitions = zeros + repetitions
        
        
        X = np.concatenate((X, p1Castle, p2Castle, turn, repetitions), axis=2)
        
        return X
    