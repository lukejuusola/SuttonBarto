import numpy as np

def softmax( prefs: np.array ) -> np.array: 
    expPref = np.exp( prefs )
    return expPref / expPref.sum()