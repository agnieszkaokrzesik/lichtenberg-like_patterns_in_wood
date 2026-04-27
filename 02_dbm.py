"""
second step: dielectric breakdown model simulation

given boundary conditions and a set of parameters, it simulates the system iteratively
by computing potentials and extending the cluster, using a potential-based criterion
"""

import numpy as np
from matplotlib import pyplot as plt
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

    # create and fill the incidence matrix
    incidence_matrix = np.zeros((edges, vertices))
    # take a node
    for v in range(vertices):
        for e in range(edges):
            # mark its neighbours
            e_left = v - 1 - v//n
            e_right = e_left + 1
            e_top = v + n*(n - 1) - n
            e_bottom = v + n*(n - 1)
            # left neighbour
            if (v % n != 0):
                incidence_matrix[e_left, v] = 1
            # right neighbour
            if (v % n != n-1):
                incidence_matrix[e_right, v] = -1 # minus marks the direction
            # top neighbour
            if v > n-1:
                incidence_matrix[e_top, v] = 1
            # bottom neighbour
            if v < n * (n-1):
                incidence_matrix[e_bottom, v] = -1 # minus marks the direction
    
    return incidence_matrix

""" clears the matrix (Lambda^T K L^-1 Lambda) to enforce BCs """
def enforce_boundaries(Lambda, K, L, coldspots, hotspots):

    # p = (Lambda^T K L^-1 Lambda)^-1 S
    bracket = np.transpose(Lambda) @ K @ np.linalg.inv(L) @ Lambda

    # enforce the BCs, by "clearing" the matrix in a proper way
    for i in np.concatenate((coldspots, hotspots)):
        bracket[i, :] = 0
        bracket[i, i] = 1

    return bracket

""" parameters """

# paths for the images and the gif
path_images = "02_images\\"
manage_directory(path_images)
path_gif = "02_dbm.gif"

# randomness control
seed = 10
random.seed(seed)
# grid size
n = 20
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

# initialise the vectors and matrices
P = np.zeros(vertices)
L = length * np.identity(edges)
K = conductivity * np.identity(edges)
Lambda = incidence_matrix(n)

# vector to mark vertices where carbon is
cluster = np.zeros(vertices)
# fill it according to BCs
cluster[hotspots] = 1

for i in range(0, 200):
    
    """ compute the potentials """ 

    # p = (Lambda^T K L^-1 Lambda)^-1 S
    bracket = enforce_boundaries(Lambda, K, L, coldspots, hotspots) 
    # solve for potentials
    Psi = np.linalg.solve(bracket, S)
    
    # plot
    plt.imshow(cluster.reshape((n, n)), interpolation='nearest', cmap='BuPu')
    plt.title(fr'cluster, iteration #{i}')
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
    if np.any(Psi[last_col] != 0): break

# animation
filenames = glob.glob(path_images + '*.png')
with imageio.get_writer(path_gif, mode='I', duration=0.05) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("your gif is ready.")

# USEFUL
# for i in range (0, n**2, n): print(sources[i:i+n])
# np.set_printoptions(threshold=np.inf)