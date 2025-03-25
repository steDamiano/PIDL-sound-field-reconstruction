import numpy as np
from scipy.sparse import diags
import cvxpy as cp

def construct_helmholtz_matrix_2d(
        n_grid_points: int, 
        k: float, 
        h: float
    ):
    '''
    Construct the 5-diagonal Toeplitz matrix Helmholtz FDM matrix. 

    Parameters
    ----------
    n_grid_points: int
        Number of points on each side of the square grid. The grid will have a total size of n_grid_points * n_grid_points.
    k: float
        Wavenumber vector.
    h: float
        Horizontal and vertical distance between adjacent pair of points.
    
    Returns
    -------
    np.ndarray: 
        Helmholtz FDM matrix
    '''

    N = n_grid_points ** 2
    
    # Values of the main and offset diagonals
    main_diag = -4 + k ** 2 * h ** 2
    off_diag = 1

    diagonals = [
        main_diag * np.ones(N),
        off_diag * np.ones(N - 1),
        off_diag * np.ones(N - 1),
        off_diag * np.ones(N - n_grid_points),
        off_diag * np.ones(N - n_grid_points)
    ]

    H = diags(diagonals, [0, -1, 1, -n_grid_points, n_grid_points], shape=(N, N)).toarray()

    return H

def frequencywise_sparse_coding(
        Y: np.ndarray,
        X: np.ndarray,
        D: np.ndarray,
        alpha: float,
        mask_idxs: np.ndarray,
    ):
    '''
    Frequency-wise sparse coding step: compute weights for a single frequency given fixed dictionary. The optimization is solved with cvxpy.
    
    Parameters
    ----------
    Y: np.ndarray 
        Input data vector, containing complex sound pressure at a single frequency on the grid.
    D: np.ndarray
        Dictionary matrix, complex-valued.
    X: np.ndarray
        Complex weights at previous iteration, for a single frequency.
    alpha: float
        Sparsity regularization parameter
    mask_idxs: np.ndarray
        Indices of available measurements on the grid.
    
    Returns
    -------
    X: np.ndarray
        Updated weights
    '''

    # Select only available measurements
    Y_masked = Y[mask_idxs]
    
    # Select corresponding dictionary rows
    D_masked = D[mask_idxs]
    
    # Definition of weight variable 
    n_components = D.shape[1]
    x = cp.Variable(n_components, complex=True, value=X)
    
    # Solution of sparse coding step using cvxpy
    objective = cp.Minimize(0.5 * cp.sum_squares(Y_masked - D_masked @ x) + alpha * cp.norm1(x))
    problem = cp.Problem(objective)
    problem.solve(solver='SCS')
    X = x.value
    
    return X

def sparse_coding(
        Y: np.ndarray,
        X: np.ndarray,
        D: np.ndarray,
        alpha: float,
        mask_idxs: np.ndarray,
    ):
    '''
    Sparse coding step: compute weights for all frequencies given fixed dictionary. The optimization is solved with cvxpy.
    
    Parameters
    ----------
    Y: np.ndarray 
        Input data vector, containing complex sound pressure in the frequency domain on the grid.
    D: np.ndarray
        Dictionary matrix, complex-valued.
    X: np.ndarray
        Complex weights at previous iteration.
    alpha: float
        Sparsity regularization parameter
    mask_idxs: np.ndarray
        Indices of available measurements on the grid.
    
    Returns
    -------
    X: np.ndarray
        Updated weights
    '''

    # Select only available measurements
    Y_masked = Y[mask_idxs]

    # Select corresponding dictionary rows
    D_masked = D[mask_idxs]

    # Definition of weight variable 
    n_components = D.shape[1]
    x = cp.Variable((n_components,Y.shape[1]), complex=True, value=X)

    # Solution of sparse coding step using cvxpy
    objective = cp.Minimize(0.5 * cp.sum_squares(Y_masked - D_masked @ x) + alpha * cp.norm1(x))
    problem = cp.Problem(objective)
    problem.solve(solver='SCS')
    X = x.value
    
    return X

def update_dictionary(
        Y: np.ndarray,
        X: np.ndarray,
        D: np.ndarray,
        beta: float,
        mask_idxs: np.ndarray,
        frequencies: np.ndarray,
        h_dist: float,
        n_grid_points: int =69
    ):
    """
    Dictionary update step: compute dictionary for current weights.
    
    Parameters
    ----------
    Y: np.ndarray
        Input data vector, containing complex sound pressure in the frequency domain on the grid.
    X: np.ndarray
        Complex weights.
    D: np.ndarray
        Dictionary matrix at previous iteration, complex-valued.
    beta: float
        Helmholtz regularization parameter
    mask_idxs: np.ndarray
        Indices of available measurements on the grid.
    h_dist: float
        Horizontal and vertical distance between adjacent pair of points.
    n_grid_points: int
    
    Returns
    -------
    D: np.ndarray
        Updated dictionary
    
    """
    # Number of dictionary atoms
    n_atoms = D.shape[1]

    # Select available pressure measurements on the grid
    Y_masked = Y[mask_idxs]
    
    # Initialize cvxpy dictionary variable
    D_new = cp.Variable(D.shape, complex=True, value=D)

    # Cost function for dictionary update step (least squares solution + Helmholtz regularization)
    def cost_function(Y_masked, X, D_new, mask_idxs, n_atoms, beta, h_dist, n_grid_points):
        helm_cost = 0
        least_squares = 0.5 * cp.sum_squares(Y_masked - D_new[mask_idxs] @ X)
        # Compute Helmholtz regularization for each dictionary atom and add to least squares cost
        for i in range(n_atoms):
            h_mat = construct_helmholtz_matrix_2d(n_grid_points, 2 * np.pi * frequencies[i] / 343, h_dist)
            helm_cost += beta * cp.norm(h_mat @ D_new[:,i]) ** 2
        
        return least_squares + helm_cost
    
    # Solve optimization problem using cvxpy
    objective = cp.Minimize(cost_function(Y_masked, X, D_new, mask_idxs, n_atoms, beta, h_dist, n_grid_points))
    problem = cp.Problem(objective)
    problem.solve(solver='SCS')
    
    # Re-normalize dictionary
    D_new = D_new.value / np.linalg.norm(D_new.value, axis=0, keepdims=True)

    return D_new