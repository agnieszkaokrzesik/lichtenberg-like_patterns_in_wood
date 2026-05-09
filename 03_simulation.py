# 2026.04.23

# rearranging the main simulation function

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
from datetime import datetime
from termcolor import colored

# """ creates or clears a directory """
# def manage_directory(path):
#     # create a folder if it doesn't exist
#     if not os.path.exists(path): os.makedirs(path)
#     # clear the folder if it exists
#     if os.path.exists(path):
#         files = glob.glob(path + '*.png')
#         for f in files: os.remove(f)

def simulation(
    cluster_colormap, T_colormap, w_colormap,
    seed, n, coldspots, hotspots,
    J, U, A, W, chi, dt, current_scaling=1):

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

    """ scales potential using the total current leaving the cluster,
    and limits the potential so that it won't exceed the maximum allowed value"""
    def scale_potential(K, L_inverse, Lambda, Psi, coldspots, J, U):

        # helper function that computes total current leaving the cluster via coldspots
        def compute_current_outflow(K, L_inverse, Lambda, Psi, coldspots):
            # currents along all edges
            I_edges = K @ L_inverse @ Lambda @ Psi
            # current in all nodes
            I_nodes = Lambda.T @ I_edges
            # total current leaving the cluster (via coldspots)
            I_out = np.sum(abs(I_nodes[coldspots]))

            return I_out

        # compute total current leaving the cluster (via coldspots)
        I_out = compute_current_outflow(K, L_inverse, Lambda, Psi, coldspots)

        # avoid division by 0
        if I_out < 1e-20: I_out = 1e-20 
        
        # potential scaling
        scale = J / I_out
        Psi *= scale

        # potential cap
        Psi_max = np.max(np.abs(Psi))
        if Psi_max > U:
            # rescale the potential
            Psi *= U / Psi_max

            print(colored("potential cap: ON", "yellow"))

        print("Psi_max", Psi_max)

        # THIS PART WAS REDUNDANT 
        # (even before this function, I_out was never used in the main loop
        # as find_new_member computes current I_edges from scratch anyways)
        # # recompute total current (using rescaled Psi)    
        # I_out = compute_current_outflow(K, L_inverse, Lambda, Psi, coldspots)

        return Psi

    """ computes temperature by solving a linear system """
    def compute_temperature(Lambda, L_inverse, Psi, sigma, A, nodes):
        # compute the "flux operator"
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

        # T = [A (Lambda.T L^-1 Lambda) + 1]^-1 Q
        term = A * operator + 1 * identity(nodes)
        T = spsolve(term, Q_nodes)

        return T

    """ computes humidity by integrating over time """
    def compute_humidity(w, T, dt):
        # function controlling drying - exponential decay
        def f(w, T):
            return w * T

        # integration
        w -= f(w, T) * np.heaviside(w, 0) * dt

        return np.maximum(w, 0) # to prevent negative w

    """ finds cluster's new member out of its neighbors using I-derived weights """
    def find_new_member(Lambda, cluster, Psi, K, L_inverse, chi, dt): 

        """ find the cluster's neighbors """
        # (term has negative values for edges that neighbor the cluster)
        term = Lambda @ cluster
        # get neighboring edges' indices
        neighbors_edges = np.where(term != 0)[0] # np.where returns a tuple

        """ compute currents """
        # compute delta_Psi along the edges
        delta_Psi = Lambda @ Psi
        # currents along the edges
        I_edges = K @ L_inverse @ delta_Psi

        """ narrow down to neighbors where |I| > 1 """
        I_abs = np.abs(I_edges[neighbors_edges])
        eligible_edges = neighbors_edges[I_abs > 1]
        # if no neighbors satisfy the condition
        if len(eligible_edges) == 0: return np.array([])

        """ draw new member(s) using current-based activation rates """
        # probability rates: r = chi I^2
        rates = chi * I_edges[eligible_edges]**2
        # P(dt) = r dt
        probabilities = rates * dt
        # draw a random threshold for each edge
        thresholds = np.random.rand(len(eligible_edges))
        # narrow down the edges again, using probabilities
        indices = np.where(probabilities > thresholds)[0]
        new_edges = eligible_edges[indices]

        """ find the node connected to the chosen edge """
        # find nodes that are connected by new_edges
        connected_nodes = Lambda[new_edges, :].nonzero()[1] # non-zero element from the chosen row; [1] to get only column indices
        # indices of the cluster elements
        cluster_nodes = np.where(cluster != 0)[0]
        # choose the ones of connected_nodes that do not belong to the cluster
        new_nodes = np.setdiff1d(connected_nodes, cluster_nodes)

        print("probab_max", max(probabilities))

        return new_nodes

    """ computes conductivity based on humidity """
    def compute_conductivity(Lambda, w, W):
        # extract the wetness of the two nodes (u, v) connected to each edge
        # (for each edge, Lambda has +1 for the source node and -1 for the target node)
        w_u = (Lambda > 0) @ w # source node
        w_v = (Lambda < 0) @ w # target node

        # compute conductivity
        # use a geometric-like product to make sigma more sensitive to low w
        w_product = w_u * w_v 
        sigma = 1 + W * w_product
        
        # update sigma
        K = diags(sigma)

        return K, sigma

    """ creates a three-panel plot visualizing cluster, temperature and humidity """
    def plot_system_state(
        cluster_colormap, T_colormap, w_colormap,
        seed, n, i, cluster, T, w,
        J, U, A, W, chi, dt, current_scaling):

        # plot initiation
        figure, cells = plt.subplots(3, 1, figsize=(5, 12)) 
        current_caption = 'current scaling' if current_scaling == 1 else 'no current scaling'
        scaling_parameters = f'\n$J = {J}$, $U = {U}$' if current_scaling == 1 else ''
        figure.suptitle(
            fr'{current_caption}, seed {seed}, $dt = {dt:.4}$, $\chi = {chi}$' + scaling_parameters
        )

        # plot - cluster
        cells[0].set_title(fr'cluster, iteration #{i}')
        cells[0].imshow(cluster.reshape((n, n)), interpolation='nearest', cmap=cluster_colormap)

        # plot - temperature
        cells[1].set_title(f'$T$, iteration #{i}, \n $A={A}$')
        cell_1 = cells[1].imshow(T.reshape((n, n)), interpolation='nearest', cmap=T_colormap)
        figure.colorbar(cell_1, ax=cells[1])

        # plot - humidity
        cells[2].set_title(f'$w$, iteration #{i}, \n $W={W}$')
        cell_2 = cells[2].imshow(w.reshape((n, n)), interpolation='nearest', cmap=w_colormap, vmin=0, vmax=1) # vmin, vmax - fixed scale
        figure.colorbar(cell_2, ax=cells[2])

        figure.tight_layout() # must be after plotting
        plt.savefig(path_images + f'simulation{i:06d}.png')
        plt.close()

    # randomness control
    random.seed(seed)
    np.random.seed(seed)

    # how often a plot is saved (for the animation), adaptive to dt
    plot_interval = 1 // (dt * 10**3)

    cap = 0 #TEMP

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
    Lambda = incidence_matrix(n)
    L = l * identity(edges)
    L_inverse = 1/l * identity(edges) # L is diagonal, so its inverse is trivial
    #K = sigma_0 * identity(edges) # removed after introducing dimensionless conductivity
    T = np.zeros(nodes)
    T[hotspots] = 1 # for burned nodes
    w = np.ones(nodes) # it starts moisturized
    K, sigma = compute_conductivity(Lambda, w, W)

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

    """ the main loop """

    for i in range(0, 2*n**2):
        
        # compute the potential
        Psi = compute_potential(Lambda, K, L, L_inverse, S, coldspots, hotspots)

        # current scaling case (direct current): rescale Psi 
        if current_scaling == 1:
            
            # scale the potential (using total current leaving the cluster)
            # after scaling, the function applies a potential cap if needed (if Psi > U)
            Psi = scale_potential(K, L_inverse, Lambda, Psi, coldspots, J, U)

        # compute the temperature
        T = compute_temperature(Lambda, L_inverse, Psi, sigma, A, nodes)

        # update humidity
        w = compute_humidity(w, T, dt)

        # visualise the cluster, temperature and humidity
        if i % plot_interval == 0:
            plot_system_state(
                cluster_colormap, T_colormap, w_colormap,
                seed, n, i, cluster, T, w,
                J, U, A, W, chi, dt, current_scaling)

        # find the cluster's new member index
        new_members = find_new_member(Lambda, cluster, Psi, K, L_inverse, chi, dt)

        print(colored(f'new members: {new_members}', "blue"))

        # update hotspots, source term, boundary conditions and the cluster
        hotspots = np.append(hotspots, new_members)
        S[hotspots] = 1
        boundary_conditions[new_members] = Psi[new_members]
        cluster[new_members] = 1

        # update conductivity
        K, sigma = compute_conductivity(Lambda, w, W)

        # terminate the loop as soon as hotspots invade the last column
        last_column = [i + (n - 1) for i in range(0, n**2, n)]
        if np.any(cluster[last_column] != 0): 
            
            # visualise the cluster, temperature and humidity (last frame)
            plot_system_state(
                cluster_colormap, T_colormap, w_colormap,
                seed, n, i, cluster, T, w,
                J, U, A, W, chi, dt, current_scaling)
            
            print(colored("potential cap used", "magenta")) if cap == 1 else print(colored("potential cup not used", "magenta"))
            
            break

""" parameters """

# paths for the images and the gif
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path_images = f"03_simulation/images_{timestamp}/"
if not os.path.exists(path_images): os.makedirs(path_images)
path_gif = f"03_simulation/animation_{timestamp}.gif"

# colors of the plot (for the cluster)
cluster_colormap = ListedColormap(["#f5c2ff", "#970B7B"])
T_colormap = 'PuRd'
w_colormap = 'RdPu'

seed = 42 # randomness control

n = 100 # grid size

# BCs: store _indices_ of nodes where Psi is 0 or > 0
hotspots =  np.arange(0, n**2, n) # potential > 0
coldspots = np.arange(n-1, n**2, n) # potential = 0

l = 1. # egde length
sigma_0 = 0.01 # basic dry system conductivity
alpha = 0.5 # thermal diffusivity
k = 1.0 # cooling constant
k_w = 8.0 # drying constant
gamma = 40.0 # humidity-conductivity relation constant
I_tot = 25. # to rescale Psi in the direct current case
I_thr = 0.000999 # to narrow down the new cluster members
V_base = 0. # baseline potential-scaling factor
V_max = 20. # to restrict scaling of the potential

#
l_0 = 1
w_0 = 1
beta = 1

# #dt = 0.5 * k_w * beta * I_thr**2 / (k * sigma_0) / 5
# #dt = 0.1 * k_w * beta * I_thr**2 / (k * sigma_0) / 5
# dt = 8e-05

# # new parameters - na razie wyliczane ze starych, żeby porównać efekt
# J = I_tot / I_thr
# U = 100 * sigma_0 * V_max / (l_0 * I_thr)
# A = alpha / (k * l_0**2)
# W = 1 * gamma * w_0**2 / sigma_0
# chi = 0.005 #k_w * beta * sigma_0 / k

#TEMP parameter set 1 (grube paprotki z 10.04)
dt = 0.5 * k_w * beta * I_thr**2 / (k * sigma_0) / 32
J = I_tot / I_thr
U = 100 * sigma_0 * V_max / (l_0 * I_thr)
A = alpha / (k * l_0**2)
W = 1 * gamma * w_0**2 / sigma_0
chi = 0.005

# #TEMP parameter set 2
# dt = 8e-5 # <- zmiana
# J = 25000
# U = 20000
# A = 0.5
# W = 4000
# chi = 0.005

# #TEMP parameter set 3
# dt = 8e-4 # <- zmiana
# J = 25000
# U = 20000
# A = 0.5
# W = 4000
# chi = 0.005

# #TEMP parameter set 4
# dt = 8e-6 # <- zmiana
# J = 25000
# U = 20000
# A = 0.5
# W = 4000
# chi = 0.005

# #TEMP parameter set 5
# dt = 8e-5 # <- zmiana znowu na -5
# J = 25000
# U = 20000
# A = 0.5
# W = 3500 # <- obniżenie
# chi = 0.005

# #TEMP parameter set 6
# dt = 8e-5
# J = 25000
# U = 20000
# A = 0.5
# W = 2000 # <- obniżenie (500 daje wyschnięcie w kilka klatek, dla 10k wilgotność prawie się nie zmienia)
# chi = 0.005

# #TEMP parameter set 7 (na bazie set 2)
# dt = 8e-5
# J = 25000
# U = 20000
# A = 1 # <--
# W = 4000
# chi = 0.005

# #TEMP parameter set 8
# dt = 8e-5
# J = 25000
# U = 20000
# A = 1
# W = 4000
# chi = 0.001 # zmniejszone x5

# #TEMP parameter set 9
# dt = 8e-5
# J = 25000
# U = 20000
# A = 1
# W = 6000
# chi = 0.0025 # zmniejszone x2

# #TEMP parameter set 10 (na bazie set 2)
# dt = 8e-5
# J = 25000
# U = 20000
# A = 10 # <---
# W = 4000
# chi = 0.005

# #TEMP parameter set 11 (na bazie set 2)
# dt = 8e-5
# J = 25000
# U = 20000
# A = 100 # <---
# W = 4000
# chi = 0.005

# #TEMP parameter set 12 (na bazie set 2)
# dt = 8e-5
# J = 25000
# U = 20000
# A = 0.001 # <---
# W = 4000
# chi = 0.005

# #TEMP parameter set 13 (set 2-based)
# dt = 8e-4 # <- zmiana
# J = 25000
# U = 20000
# A = 0.5
# W = 20000
# chi = 0.001

# #TEMP parameter set 14 (set 2-based)
# dt = 8e-4 # <- zmiana
# J = 2500
# U = 20000
# A = 1.
# W = 20000
# chi = 0.001

# #TEMP parameter set 15 (set 2-based)
# dt = 8e-4
# J = 2500  # <- zmiana
# U = 20000
# A = 1.
# W = 20000
# chi = 0.001

# #TEMP parameter set 16 (set 2-based)
# dt = 8e-4
# J = 2500  # <- zmiana
# U = 15000
# A = 1.
# W = 17000
# chi = 0.001

print("J, U, A, W, chi", J, U, A, W, chi)

# run the simulation
simulation(
    cluster_colormap, T_colormap, w_colormap,
    seed, n, coldspots, hotspots,
    J, U, A, W, chi, dt, current_scaling=1)

# make the animation
filenames = glob.glob(path_images + '*.png')
with imageio.get_writer(path_gif, mode='I', fps=12, dpi=200) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("the gif is ready.")

# USEFUL printing
# for i in range (0, n**2, n): print(sources[i:i+n])
# np.set_printoptions(threshold=np.inf)
