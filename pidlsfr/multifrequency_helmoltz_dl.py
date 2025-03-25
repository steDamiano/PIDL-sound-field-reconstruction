import numpy as np
import matplotlib.pyplot as plt
import os

from utils import default_soundfield_config, Soundfield,sample_surface, sinc_kernel_multifreq
from pidl_core import construct_helmholtz_matrix_2d, sparse_coding, frequencywise_sparse_coding, update_dictionary
from metrics import nmse, ncc

SEED = 42
np.random.seed(SEED)

if __name__ == '__main__':
    n_grid_points = 69          # Number of points on each side of the square grid
    h_dist = 0.025              # Distance between pairs of adjacent points
    M = n_grid_points ** 2      # Total number of points in the grid
    N = 50                      # Number of microphones (i.e., available measurements on the grid)
    f_min = 500                 # Lower limit of chosen frequency range
    f_max = 700                 # Upper limit of chosen frequency range
    Nf = 20 + 1                 # Number of training frequencies in chosen range
    Nf_rec = 80 + 1             # Number of reconstruction frequencies in chosen range

    # Dictionary learning parameters
    max_iters = 20              # Maximum training iterations
    K = Nf                      # Number of dictionary atoms
    tol = 5e-02                 # Tolerance for early training stopping
    alpha = 1                   # Sparsity regularization parameter
    beta = 0.1                  # Helmholtz regularization parameter
    
    frequencies = np.linspace(f_min,f_max,Nf)
    frequencies_rec = np.linspace(f_min,f_max,Nf_rec)
    
    # Initialize measurements matrix - full grid
    Y = np.zeros((M,Nf), dtype=np.complex64)

    # Inizialize measurements matrix - available measurements
    Y_masked = np.zeros_like(Y)
    
    # Randomly select N microphones on the grid (indices in the measurement matrix are picked to create a mask)
    field_opts = default_soundfield_config(measurement='019', frequency=600)
    sfo = Soundfield(**field_opts)
    indexes = sample_surface((69,69), N, 'sample', sfo.r, 0.07, SEED)
    mask = np.zeros(M)
    mask[indexes] = 1
    
    # Load training pressure
    for i in range(Nf):
        field_opts = default_soundfield_config(measurement='019', frequency=frequencies[i])
        sfo = Soundfield(**field_opts)
        Y[:,i] = sfo._pm.squeeze()      # Full grid
        Y_masked[:,i] = Y[:,i] * mask   # Selected microphones

    # Load evaluation pressure
    Y_rec = np.zeros((M,Nf_rec), dtype=np.complex64)
    Y_rec_masked = np.zeros_like(Y_rec)
    for i in range(Nf_rec):
        field_opts = default_soundfield_config(measurement='019', frequency=frequencies_rec[i])
        sfo = Soundfield(**field_opts)
        Y_rec[:,i] = sfo._pm.squeeze()          # Full grid
        Y_rec_masked[:,i] = Y_rec[:,i] * mask   # Selected microphones

    # Initialize dictionary, corresponding to BL dictionary
    if os.path.exists(f'sinc_dictionary_{K}_atoms_{f_min}-{f_max}.npy'):
        D_dl = np.load(f'sinc_dictionary_{K}_atoms_{f_min}-{f_max}.npy')
    else:
        D_dl = sinc_kernel_multifreq((n_grid_points, n_grid_points), K, frequencies, h_dist)
    
    D_dl = 1 / np.linalg.norm(D_dl, axis=0, keepdims=True) * D_dl
    
    # Initialize weights -> random initialization
    X_dl = np.random.randn(K,Nf) + 1j * np.random.randn(K,Nf)
    X_dl /= np.linalg.norm(X_dl, axis=0)

    past_cost = np.inf
    # Training loop
    for i in range(max_iters):
        # Dictionary learning step (sparse coding + dictionary update)
        X_dl = sparse_coding(Y, X_dl, D_dl, alpha, indexes)
        D_dl = update_dictionary(Y, X_dl, D_dl, beta, indexes, frequencies, h_dist=h_dist, n_grid_points=n_grid_points)

        # Compute cost function: least squares + sparsity regularization + Helmholtz regularization (one term per each dictionary atom)
        current_cost = 0.5 * np.sum(np.abs(Y_masked[indexes] - D_dl[indexes] @ X_dl) ** 2) + alpha * np.linalg.norm(X_dl,1)
        for k in range(K):
            h_mat = construct_helmholtz_matrix_2d(n_grid_points, 2 * np.pi * frequencies[k] / 343, h_dist)
            current_cost += beta * np.linalg.norm(h_mat @ D_dl[:,k]) ** 2
        
        print(f"Iter {i+1} of {max_iters}, current_cost: {current_cost}")
        if np.abs(current_cost - past_cost) < tol:
            print('Converged, cost changed by ', np.abs(current_cost - past_cost))
            break
        past_cost = current_cost
    
    print('='*50)

    X_dl_perfreq = np.zeros((K,Nf_rec), dtype=np.complex64)
    nmse_perfreq = np.zeros(Nf_rec)
    ncc_perfreq = np.zeros(Nf_rec)
    
    # Evaluation loop --> frequency-wise
    for f in range(Nf_rec):
        
        # Compute weights for frequency f using learnt dictionary
        X_dl_perfreq[:,f] = frequencywise_sparse_coding(Y_rec[:,f], X_dl_perfreq[:,f], D_dl, alpha, indexes)
        y_hat_dl = D_dl @ X_dl_perfreq[:,f]
        
        # Compute metrics
        nmse_perfreq[f] = nmse(Y_rec[:,f], y_hat_dl)
        ncc_perfreq[f] = ncc(Y_rec[:,f], y_hat_dl)
        
        print(f'Sparse coding, freq: {frequencies_rec[f]} - nonzeros: {len(X_dl_perfreq[np.abs(X_dl_perfreq[:,f] > 1e-07),f])} - nmse: {nmse_perfreq[f]}')

        # Optional: plot sound field (uncomment)
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(np.reshape(np.abs(Y_rec[:,f]), (n_points, n_points)))
        # plt.title(f'Ground truth - {frequencies_rec[f]} Hz')

        # plt.subplot(1,2,2)
        # plt.imshow(np.reshape(np.abs(y_hat_dl), (n_points, n_points)))
        # plt.title(f'Frequency: {frequencies_rec[f]} - nmse: {nmse_perfreq[f]}')
        # plt.savefig(f'figures/field_{str(frequencies_rec[f])}_proposed.png')

    print(f'Average NMSE: {np.mean(nmse_perfreq)}')
    print(f'Average NCC: {np.mean(ncc_perfreq)}')
    