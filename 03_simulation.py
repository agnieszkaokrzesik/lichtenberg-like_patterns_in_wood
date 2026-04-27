# 2025.11.19 version

# first (buggy) version of the water-introducing simulation

# this code is a version developed from a previous 02_dbm.py variant that is not included in this repository, and introduces the following changes:
# * clearer variable names and general clarity improvements,
# * introduction of new physical variables and parameters to account for drying,
# * computation of temperature, humidity and conductivity (T, w, K & sigma),
# * a three-panel plot of the system.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.sparse import csr_matrix, identity, diags
from scipy.sparse.linalg import spsolve
import random
import os
import glob
import imageio.v2 as imageio

""" creates or clears a directory """
def manage_directory(path):
    # create a folder if it doesn't exist
    if not os.path.exists(path): os.makedirs(path)
    # clear the folder if it exists
    if os.path.exists(path):
        files = glob.glob(path + '*.png')
        for f in files: os.remove(f)

""" creates the incidence matrix for an nxn grid """
def incidence_matrix(n):

    nodes = n**2
    edges = 2*n * (n - 1)

    # arrays to store data needed to construct a sparse matrix
    rows = []
    cols = []
    values = []

    # helper function to save data for the sparse matrix
    def save_for_sparse(row, col, value):
        rows.append(row)
        cols.append(col)
        values.append(value)

    # take a node
    for v in range(nodes):
        # check its neighbors:
        # left neighbor
        if (v % n != 0):
            e_left = v - 1 - v//n
            # save the coordinates and the value for the sparse matrix
            save_for_sparse(e_left, v, 1)
        # right neighbor
        if (v % n != n-1):
            e_right =  v - v//n
            save_for_sparse(e_right, v, -1) # minus marks the direction
        # top neighbor
        if v > n-1:
            e_top = v + n*(n - 1) - n
            save_for_sparse(e_top, v, 1)
        # bottom neighbor
        if v < n * (n-1):
            e_bottom = v + n*(n - 1)
            save_for_sparse(e_bottom, v, -1) # minus marks the direction
    
    incidence_matrix = csr_matrix((values, (rows, cols)), shape=(edges, nodes))

    return incidence_matrix

""" clears the (Lambda^T K L^-1 Lambda) matrix to enforce BCs """
def enforce_boundary_conditions(Lambda, K, L, L_inverse, coldspots, hotspots):

    # p = (Lambda^T K L^-1 Lambda)^-1 S
    bracket = np.transpose(Lambda) @ K @ L_inverse @ Lambda

    # create and fill the helper vector
    zeroing_vector = np.ones(bracket.shape[0])
    zeroing_vector[np.concatenate((hotspots, coldspots))] = 0
    # reshape it for multiplication
    zeroing_vector_reshaped = zeroing_vector[:, np.newaxis]

    # multiply the matrix element-wise by enforcing_vector to zero out the desired rows
    bracket = bracket.multiply(zeroing_vector_reshaped)

    # set the diagonal elements corresponding to hot/coldspots to 1 to properly enforce BCs
    bracket += diags(1 - zeroing_vector)

    return bracket

""" parameters """

# paths for the images and the gif
path_images = "03_images\\"
manage_directory(path_images)
path_gif = "03_simulation.gif"

# colors of the plot
colors = ListedColormap(["#f5c2ff", "#970B7B"])

n = 20 # grid size
length = 1 # egde length
sigma_0 = 0.005 # basic dry system conductivity
alpha = 0.05 # thermal diffusivity
k = 50 # cooling constant
k_w = 10 # drying constant
gamma = 0.45 # humidity-conductivity relation constant
dt = 0.01

# BCs: store _indices_ of nodes where Psi is 0 or > 0
hotspots =  np.arange(0, n**2, n) # potential > 0
coldspots = np.arange(n-1, n**2, n) # potential = 0

""" initialisation """

nodes = n**2
edges = 2*n * (n - 1)

# source term, which marks the hotspots
S = np.zeros(nodes)
S[hotspots] = 1

# vector containing _values_ of BCs 
boundary_conditions = np.zeros(nodes)
boundary_conditions[hotspots] = 1

# initialise the vectors and sparse matrices
L = length * identity(edges)
L_inverse = 1/length * identity(edges) # L is diagonal, so its inverse is trivial
K = sigma_0 * identity(edges)
T = np.zeros(nodes)
T[hotspots] = 1 # for burned nodes
w = np.ones(nodes) # it starts moisturized
sigma = np.full(edges, sigma_0)
Lambda = incidence_matrix(n)

# vector to mark nodes where carbon is
cluster = np.zeros(nodes)

# indexes of all the boundary nodes
boundary_nodes = np.unique(np.concatenate((
    np.arange(0, n),
    np.arange(n*(n-1), n**2),
    np.arange(0, n**2, n),
    np.arange(n-1, n**2, n))))

for i in range(0, n**2):
    
    """ compute the potentials """ 

    # Psi = (Lambda^T K L^-1 Lambda)^-1 S
    bracket = enforce_boundary_conditions(Lambda, K, L, L_inverse, coldspots, hotspots) 
    # solve for potentials
    Psi = spsolve(bracket, S)

    # plot - potential
    figure, cells = plt.subplots(3, 1, figsize=(5, 12))
    figure.suptitle(fr"$\sigma_0 = {sigma_0}, \gamma = {gamma}$")
    cells[0].set_title(fr'cluster, iteration #{i}')
    cells[0].imshow(cluster.reshape((n, n)), interpolation='nearest', cmap=colors)

    """ find the cluster's new member """

    # mark the carbon nodes
    cluster[hotspots] = 1

    # find the cluster's neighbors
    neighbors_mask = np.transpose(Lambda) @ Lambda @ cluster
    # matrix containing 1 at the indices of neighbors
    neighbors_mask = (neighbors_mask < 0).astype(int)
    neighbors_indices = np.where(neighbors_mask == 1)[0]
    # potentials of neighbors
    Psi_border = neighbors_mask * Psi

    # draw the new cluster member based on Psi
    weights = (1 - Psi_border[neighbors_indices])**2
    new_member = random.choices(neighbors_indices, weights=weights)

    # update hotspots, source term, boundary conditions and the cluster
    hotspots = np.append(hotspots, new_member)
    S[hotspots] = 1
    boundary_conditions[new_member] = Psi[new_member]
    cluster[new_member] = 1

    """ compute the temperature """

    # compute the "flux operator" (we don't compute the whole J as it contains T we're looking for)
    # J = -alpha (Lambda.T L^-1 Lambda) T
    operator = Lambda.T @ L_inverse @ Lambda

    # compute the heat
    # Q = 1/2 Lambda.T sigma (L^-1 Lambda Psi)^2
    delta_Psi = Lambda @ Psi
    bracket = L_inverse @ delta_Psi
    Q_edges = sigma * bracket**2
    Q_nodes = 0.5 * abs(Lambda).T @ Q_edges

    # T = [alpha (Lambda.T L^-1 Lambda) + k 1]^-1 Q
    term = alpha * operator + k * identity(nodes)
    T = spsolve(term, Q_nodes)
    #print(T)

    # plot - temperature
    cells[1].set_title(fr'$T$, iteration #{i}, $k=${k}, $\alpha=${alpha}')
    cell_1 = cells[1].imshow(T.reshape((n, n)), interpolation='nearest', cmap="PuRd")
    figure.colorbar(cell_1, ax=cells[1])

    """ update humidity """

    # function controlling drying
    def f(T):
        return k_w * T

    # integration
    w -= f(T) * np.heaviside(w, 0) * dt

    # plot - humidity
    cells[2].set_title(fr'$w$, iteration #{i}, $k_w=${k_w}')
    cell_2 = cells[2].imshow(w.reshape((n, n)), interpolation='nearest', cmap="Purples")
    figure.colorbar(cell_2, ax=cells[2])
    figure.tight_layout()
    plt.savefig(path_images + f'simulation{i:06d}.png')
    plt.close()

    """ update conductivity """

    # compute condictivity
    # sigma = sigma_0 + 1/2 gamma Lambda_absolute w
    w_average = 0.5 * abs(Lambda) @ w
    sigma = sigma_0 + gamma * w_average
    # update sigma
    K = diags(sigma)

    # terminate the loop as soon as hotspots invade the last column
    last_column = [i + (n - 1) for i in range(0, n**2, n)]
    if np.any(Psi[last_column] != 0): break

# animation
filenames = glob.glob(path_images + '*.png')
with imageio.get_writer(path_gif, mode='I', fps=24, dpi=200) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("your gif is ready.")

# USEFUL
# for i in range (0, n**2, n): print(sources[i:i+n])
# np.set_printoptions(threshold=np.inf)