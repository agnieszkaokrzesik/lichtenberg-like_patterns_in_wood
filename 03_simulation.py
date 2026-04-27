# 2026.03.02 version

# cluster growth bug fixes

# changes:
# * bug fixes in find_new_member,
# * plot title update to reflect current-based growth.

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
from termcolor import colored

""" creates or clears a directory """
def manage_directory(path):
    # create a folder if it doesn't exist
    if not os.path.exists(path): os.makedirs(path)
    # clear the folder if it exists
    if os.path.exists(path):
        files = glob.glob(path + '*.png')
        for f in files: os.remove(f)

""" returns incidence matrix for an nxn grid """
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

""" returns cleared version of a matrix used to enforce BCs """
def term_BCs_enforced(Lambda, K, L, L_inverse, coldspots, hotspots):

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

""" computes potential by solving a linear system (with enforced BCs) """
def compute_potential(Lambda, K, L, L_inverse, S, coldspots, hotspots):
    # Psi = (Lambda^T K L^-1 Lambda)^-1 S
    bracket = term_BCs_enforced(Lambda, K, L, L_inverse, coldspots, hotspots) 
    # solve for potentials
    Psi = spsolve(bracket, S)

    return Psi

""" finds cluster's new member out of its neighbors using I-derived weights """
def find_new_member(Lambda, cluster, Psi, sigma, threshold): 

    """ find the cluster's neighbors """
    # (term has negative values for edges that neighbor the cluster)
    term = Lambda @ cluster
    # get neighboring edges' indices
    neighbors_edges = np.where(term != 0)[0] # np.where returns a tuple

    """ compute current densities """
    # compute delta_Psi along the edges
    delta_Psi = Lambda @ Psi
    # compute current density (caution: * instead of @ needed)
    j = sigma * delta_Psi

    """ choose the new cluster member """
    # narrow down to neighbors where j > threshold
    j_abs = np.abs(j[neighbors_edges])
    narrowed_edges = neighbors_edges[j_abs > threshold]
    # if no neighbors satisfy the condition
    if len(narrowed_edges) == 0: return None
    # draw the new cluster member based on j^2
    weights = j[narrowed_edges]**2
    new_edge = random.choices(narrowed_edges, weights=weights)[0] # random.choices returns a list (one-item-long here)

    print(colored(f"j[narrowed_edges]: {j[narrowed_edges]}", "magenta"))
    print("% of edges passing thr", len(narrowed_edges) / len(neighbors_edges))
    print(f"max |j|: {max(abs(j))}")

    """ find the node connected to the chosen edge """
    # find nodes that are connected by new_edge
    connected_nodes = Lambda[new_edge, :].nonzero()[1] # non-zero element from the chosen row; [1] to get only column indices
    # indices of the cluster elements
    cluster_nodes = np.where(cluster != 0)
    # choose the one of connected_nodes that is not the member of the cluster
    new_node = np.setdiff1d(connected_nodes, cluster_nodes)

    return new_node

""" computes temperature by solving a linear system """
def compute_temperature(Lambda, L_inverse, Psi, sigma, alpha, k, nodes):
    # compute the "flux operator" (we don't compute the whole J as it contains T we're looking for)
    # J = -alpha (Lambda.T L^-1 Lambda) T
    operator = Lambda.T @ L_inverse @ Lambda

    # compute the heat
    # Q = 1/2 Lambda.T sigma (L^-1 Lambda Psi)^2
    delta_Psi = Lambda @ Psi
    bracket = L_inverse @ delta_Psi
    # heat along all edges
    Q_edges = sigma * bracket**2
    # heat in the nodes, with 0.5 factor as each edge heats two nodes
    Q_nodes = 0.5 * abs(Lambda).T @ Q_edges

    # T = [alpha (Lambda.T L^-1 Lambda) + k 1]^-1 Q
    term = alpha * operator + k * identity(nodes)
    T = spsolve(term, Q_nodes)

    return T

""" computes humidity by integrating over time """
def compute_humidity(w, T, k_w, dt):
    # function controlling drying
    def f(T):
        return k_w * T

    # integration
    w -= f(T) * np.heaviside(w, 0) * dt
    print(colored(f"T: {T}", "red"))
    print(colored(f"w: {w}", "blue"))

    return np.maximum(w, 0) # to prevent negative w

""" computes conductivity based on humidity """
def compute_conductivity(Lambda, w, sigma_0, gamma):
    # compute condictivity
    # sigma = sigma_0 + 1/2 gamma Lambda_absolute w
    w_average = 0.5 * abs(Lambda) @ w
    sigma = sigma_0 + gamma * w_average**2 #01.27 ZMIANA, dodanie **2
    # update sigma
    K = diags(sigma)
    #print(colored(w, "cyan"))

    return K, sigma

""" parameters """

# paths for the images and the gif
path_images = "03_images\\"
manage_directory(path_images)
path_gif = "03_simulation.gif"

# ############################### TEMP
# path2 = "03_psi\\"
# manage_directory(path2)
# ###############################

# colors of the plot (for the cluster)
colors = ListedColormap(["#f5c2ff", "#970B7B"])

n = 100 # grid size
length = 1 # egde length
sigma_0 = 0.05 # basic dry system conductivity
alpha = 1 # thermal diffusivity
k = 0.1 # cooling constant
k_w = .75 # drying constant
gamma = 1 # humidity-conductivity relation constant
I_target = 1 # to rescale Psi in the direct current case
threshold = 0.001 # to narrow down the new cluster members
dt = 1

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
# initialise the cluster according to BCs
cluster[hotspots] = 1

# indexes of all the boundary nodes
boundary_nodes = np.unique(np.concatenate((
    np.arange(0, n),
    np.arange(n*(n-1), n**2),
    np.arange(0, n**2, n),
    np.arange(n-1, n**2, n))))

def simulation(
    n, Lambda, K, L, L_inverse, sigma,
    S, coldspots, hotspots, cluster,
    alpha, k, w, k_w, dt, sigma_0, gamma,
    threshold, I_target, alternating_current=1):

    for i in range(0, n**2):
        
        # compute the potential
        Psi = compute_potential(Lambda, K, L, L_inverse, S, coldspots, hotspots)

        # direct current case: rescale Psi 
        if alternating_current != 1:
            # currents along all edges
            I = K @ L_inverse @ Lambda @ Psi
            # indices of edges coming into the electrode
            # (horizontal edges connecting the last two columns)
            right_slice = np.arange(n - 2, n - 1 + (n - 1)**2, n - 1)
            # total current in the rightmost column
            I_right = np.sum(I[right_slice])
            # rescaling
            Psi = Psi * (I_target / I_right)

            # # righ_slice check
            # test_slice = np.arange(n - 3, n + (n - 1)**2, n - 1)
            # I_test = np.sum(I[test_slice])
            # print(colored(right_slice, "blue"))
            # print(colored(I_right, "blue"))
            # print(colored(test_slice, "yellow"))
            # print(colored(I_test, "yellow"))
        
        # TEMP print(colored(I, "cyan"))

        # ##################################################### TEMP (psi plots)
        # delta_Psi = Lambda @ Psi
        # gradient_Psi = abs(Lambda.T @ delta_Psi) 

        # term = Lambda.T @ Lambda @ cluster
        # neighbors_indices = np.where(term < 0)[0] # np.where returns a tuple
        
        # p, c = plt.subplots(2, 2, figsize=(18, 14))
        # x = np.arange(0, neighbors_indices.shape[0], 1)
        # # gradPsi
        # c[0,0].scatter(x, gradient_Psi[neighbors_indices], label="gradPsi[cluster_neighbors]")
        # c[0,0].plot(x, [threshold]*len(x), label=f'threshold')
        # c[0,0].set_xlabel("idx")
        # c[0,0].legend()
        # # gradPsi colormap
        # c01 = c[0,1].imshow(gradient_Psi.reshape((n, n)), interpolation='nearest', cmap="Blues")
        # c[0,1].set_title("gradPsi")
        # p.colorbar(c01, ax=c[0,1])
        # # Psi
        # c[1,0].scatter(x, Psi[neighbors_indices], label="Psi[cluster_neighbors]")
        # c[1,0].set_xlabel("idx")
        # c[1,0].legend()
        # # Psi colormap
        # c11 = c[1,1].imshow(Psi.reshape((n, n)), interpolation='nearest', cmap="Blues")
        # c[1,1].set_title("Psi")
        # p.colorbar(c11, ax=c[1,1])
        # p.savefig(path2 + f'simulation{i:06d}.png')
        # plt.close()
        # #######################################################

        # plot
        figure, cells = plt.subplots(3, 1, figsize=(5, 12)) 
        current_caption = f'direct current, $I=${I_target}' if alternating_current != 1 else 'alternating current'
        figure.suptitle(
            f'{current_caption} \n'
            f'new members only if $|\,j\,| > {threshold}$'
        )
        # plot - cluster
        cells[0].set_title(fr'cluster, iteration #{i}')
        cells[0].imshow(cluster.reshape((n, n)), interpolation='nearest', cmap=colors)

        # find the cluster's new member index
        new_member = find_new_member(Lambda, cluster, Psi, sigma, threshold)
        # stop the simulation if there's no new member
        if new_member == None:
            print("no valid new member - simulation stopped.")
            break

        # update hotspots, source term, boundary conditions and the cluster
        hotspots = np.append(hotspots, new_member)
        S[hotspots] = 1
        boundary_conditions[new_member] = Psi[new_member]
        cluster[new_member] = 1

        # compute the temperature
        T = compute_temperature(Lambda, L_inverse, Psi, sigma, alpha, k, nodes)

        # plot - temperature
        cells[1].set_title(f'$T$, iteration #{i}, \n $k=${k}, $\\alpha=${alpha}')
        cell_1 = cells[1].imshow(T.reshape((n, n)), interpolation='nearest', cmap="PuRd")
        figure.colorbar(cell_1, ax=cells[1])

        # update humidity
        w = compute_humidity(w, T, k_w, dt)

        #np.set_printoptions(threshold=np.inf)
        #print(colored(w[np.where(cluster != 0)], "blue")) # TEMP

        # plot - humidity
        cells[2].set_title(f'$w$, iteration #{i}, \n $k_w=${k_w}, $dt=${dt}, $\\sigma_0=${sigma_0}, $\\gamma=${gamma}')
        cell_2 = cells[2].imshow(w.reshape((n, n)), interpolation='nearest', cmap="Purples", vmin=0, vmax=1) # vmin, vmax - fixed scale
        figure.colorbar(cell_2, ax=cells[2])

        figure.tight_layout() # must be after plotting
        plt.savefig(path_images + f'simulation{i:06d}.png')
        plt.close()

        # update conductivity
        K, sigma = compute_conductivity(Lambda, w, sigma_0, gamma)
        # TEMP print(colored(sigma, "magenta"))
        # (no plot as conductivity is defined in the edges)

        # terminate the loop as soon as hotspots invade the last column
        last_column = [i + (n - 1) for i in range(0, n**2, n)]
        if np.any(cluster[last_column] != 0): break

# run the simulation
simulation(
    n, Lambda, K, L, L_inverse, sigma,
    S, coldspots, hotspots, cluster,
    alpha, k, w, k_w, dt, sigma_0, gamma,
    threshold, I_target, alternating_current=0)

# make the animation
filenames = glob.glob(path_images + '*.png')
with imageio.get_writer(path_gif, mode='I', fps=24, dpi=200) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("the gif is ready.")

# USEFUL
# for i in range (0, n**2, n): print(sources[i:i+n])
# np.set_printoptions(threshold=np.inf)