import numpy as np
from librosa.core import lpc

def librosa_lpc(X, order):
    try:
        return lpc(X, order)
    except:
        res = np.zeros((order + 1,))
        res[0] = 1.
        return res