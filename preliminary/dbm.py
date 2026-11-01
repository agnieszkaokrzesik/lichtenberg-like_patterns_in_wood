"""
second step: dielectric breakdown model simulation - sparse matrix version

given boundary conditions and a set of parameters, it simulates the system iteratively
by computing potentials and extending the cluster, using a potential-based criterion
and sparse matrix formulations
"""

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

    vertices = n**2
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
    for v in range(vertices):
        # check its neighbours:
        # left neighbour
        if (v % n != 0):
            e_left = v - 1 - v//n
            # save the coordinates and the value for the sparse matrix
            save_for_sparse(e_left, v, 1)
        # right neighbour
        if (v % n != n-1):
            e_right =  v - v//n
            save_for_sparse(e_right, v, -1) # minus marks the direction
        # top neighbour
        if v > n-1:
            e_top = v + n*(n - 1) - n
            save_for_sparse(e_top, v, 1)
        # bottom neighbour
        if v < n * (n-1):
            e_bottom = v + n*(n - 1)
            save_for_sparse(e_bottom, v, -1) # minus marks the direction
    
    incidence_matrix = csr_matrix((values, (rows, cols)), shape=(edges, vertices))

    return incidence_matrix

""" clears the (Lambda^T K L^-1 Lambda) matrix to enforce BCs """
def enforce_boundaries(Lambda, K, L, L_inverse, coldspots, hotspots):

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
path_images = "02_images\\"
manage_directory(path_images)
path_gif = "02_dbm.gif"

# plot colors
colors = ListedColormap(["#c2d4ff", "#150B53"])

# randomness control
seed = 505
random.seed(seed)

# grid size
n = 100
# constants
length = 1
conductivity = 1

# BCs: store _indices_ of vertices where Psi is 0 or > 0
hotspots =  np.arange(0, n**2, n) # potential > 0
coldspots = np.arange(n-1, n**2, n) # potential = 0

""" initialisation """

vertices = n**2
edges = 2*n * (n - 1)

# source term, which marks the hotspots
S = np.zeros(vertices)
S[hotspots] = 1

# vector containing _values_ of BCs 
boundaries = np.zeros(vertices)
boundaries[hotspots] = 1

# initialise the vectors and sparse matrices
P = np.zeros(vertices)
L = length * identity(edges)
L_inverse = 1/length * identity(edges) # L is diagonal, so its inverse is trivial
K = conductivity * identity(edges)
Lambda = incidence_matrix(n)

# initiate the figure
plt.figure(dpi=150, figsize=(6, 6))

# vector to mark vertices where carbon is
cluster = np.zeros(vertices)
# fill it according to BCs
cluster[hotspots] = 1

for i in range(0, n**2):
    
    """ compute the potentials """ 

    # p = (Lambda^T K L^-1 Lambda)^-1 S
    bracket = enforce_boundaries(Lambda, K, L, L_inverse, coldspots, hotspots) 
    # solve for potentials
    Psi = spsolve(bracket, S)

    # plot
    if i % 10 == 0:
        plt.imshow(cluster.reshape((n, n)), interpolation='nearest', cmap=colors)
        plt.title(fr'cluster, seed {seed}, iteration #{i}')
        plt.savefig(path_images + f'img{i:06d}.png')
        # plt.show()

    """ find the cluster's new member """

    # mark the carbon vertices
    cluster[hotspots] = 1

    # find the cluster's neighbours
    A = np.transpose(Lambda) @ Lambda @ cluster
    # matrix containing 1 at the indices of neighbours
    A = (A < 0).astype(int)
    neighbours = np.where(A == 1)[0]
    # potentials of neighbours
    Psi_border = A * Psi

    # draw the new cluster member based on Psi
    weights = (1 - Psi_border[neighbours])**2
    new_member = random.choices(neighbours, weights=weights)

    # update hotspots, source term, boundaries and the cluster
    hotspots = np.append(hotspots, new_member)
    S[hotspots] = 1
    boundaries[new_member] = Psi[new_member]
    cluster[new_member] = 1

    # terminate the loop as soon as hotspots invade the last column
    last_col = [i + (n - 1) for i in range(0, n**2, n)]
    if np.any(Psi[last_col] != 0): 
        # plot the last frame
        plt.imshow(cluster.reshape((n, n)), interpolation='nearest', cmap=colors)
        plt.title(fr'cluster, seed {seed}, iteration #{i}')
        plt.savefig(path_images + f'img{i:06d}.png')
        
        break

# animation
filenames = glob.glob(path_images + '*.png')
with imageio.get_writer(path_gif, mode='I', fps=24) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("your gif is ready.")

# USEFUL
# for i in range (0, n**2, n): print(sources[i:i+n])
# np.set_printoptions(threshold=np.inf)