"""
first simple step in developing the target simulation

given a source term, it computes the potential distribution in the entire
system with use of Ohm's law and an incidence matrix
"""

import numpy as np
from matplotlib import pyplot as plt

# parameters
n = 6
length = 1
conductivity = 1

vertices = n**2
edges = 2*n * (n - 1)

# initialise the vectors and matrices
potentials = np.zeros(vertices)
length_matrix = length * np.identity(edges)
conductivity_matrix = conductivity * np.identity(edges)

# create and fill the source term
sources = np.zeros(vertices)
sources[::n] = 1 # left column
sources[19] = 1 # one additional point

# create and fill the incidence matrix
incidence_matrix = np.zeros((edges, vertices))
# take a node
for v in range(vertices):
    for e in range(edges):
        # mark its neighbours
        e_left = v - 1 - v//n
        e_right = e_left + 1
        e_top = v + (n - 1) * n - n
        e_bottom = v + (n - 1) * n
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

# compute the potentials
# p = (Delta^T K L^-1 Delta)^-1 S
factor = np.transpose(incidence_matrix) @ conductivity_matrix @ np.linalg.inv(length_matrix) @ incidence_matrix

# clear the 'factor' matrix to satisfy BCs
# potential = 0
for i in range (n-1, n**2, n):
    factor[i, :] = 0
    factor[i, i] = 1
# potential = 1
for i in range (0, n**2, n):
    factor[i, :] = 0
    factor[i, i] = 1
factor[19, :] = 0
factor[19, 19] = 1

# for i in range (0, n**2, n): print(sources[i:i+n])
# np.set_printoptions(threshold=np.inf)

# solve for potentials
potentials = np.linalg.solve(factor, sources)

# the result
print("potentials:")
print(potentials)
plt.title(fr'$\Psi$')
plt.imshow(potentials.reshape((n, n)), interpolation = 'nearest', cmap = 'RdPu')
plt.colorbar()
plt.show()

