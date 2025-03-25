import numpy as np

def nmse(y: np.ndarray, x: np.ndarray):
    '''
    Normalized mean squared error metric in dB scale.
    
    Parameters
    ----------
    y: np.ndarray
        Ground truth pressure vector
    x: np.ndarray
        Estimated pressure vector
    
    Returns
    -------
    float
        Normalized mean squared error between x and y
    '''

    return 20 * np.log10(np.linalg.norm(x.ravel() - y.ravel())) - 20 * np.log10(np.linalg.norm(y.ravel()))

def ncc(y: np.ndarray, x: np.ndarray):
    '''
    Normalized cross-correlation metric
    
    Parameters
    ----------
    y: np.ndarray
        Ground truth pressure vector
    x: np.ndarray
        Estimated pressure vector
    Returns
    -------
    float
        Normalized cross correlation between x and y
    '''
    x = x.ravel()
    y = y.ravel()

    return np.abs(x.ravel() @ np.conj(y.ravel())) / (np.linalg.norm(x.ravel()) * np.linalg.norm(y.ravel()))