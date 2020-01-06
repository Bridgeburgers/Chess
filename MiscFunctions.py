import numpy as np
#%%
def RandomArgSort(a, reverse=True):
    """
    Give the argsort elements of a numeric vector with a random tiebreaker
    """
    b = np.random.random(len(a))
    args = np.lexsort((b,a))
    if reverse:
        return args[::-1]
    else:
        return args
    
def Softmax(v, temp=1):
    expv = np.exp(v * temp)
    return expv / np.sum(expv)