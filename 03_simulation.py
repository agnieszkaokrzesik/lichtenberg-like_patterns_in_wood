# reviewed version (see 2026.03.22 email)

# exponential drying, geometric mean in conductivity formula, new current scaling

# changes:
# * random seed addition,
# * parameter adjustment,
# * exponential decay in drying: f(T, w) =  k_w * T * w,
# * geometric mean in conductivity formula to block nodes more aggressively,
# * current scaling by I[coldspots] (instead of I[right_slice]),
# * plotting every 10 iterations.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.sparse import csr_matrix, identity, diags
from scipy.sparse.linalg import spsolve
import random
random.seed(42)
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
    term = Lambda @ cluster
    neighbors_edges = np.where(term != 0)[0] 

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
    w -= (k_w * T) * np.heaviside(w, 0) * w * dt
#    print(colored(f"T: {T}", "red"))
#    print(colored(f"w: {w}", "blue"))

    return np.maximum(w, 0) # to prevent negative w

""" computes conductivity based on humidity """

def compute_conductivity(Lambda, w, sigma_0, gamma):
    # Extract the wetness of the two nodes connected to each edge.
    # Lambda has a 1 (source) and a -1 (target) for each edge.
    w_u = (Lambda > 0) @ w
    w_v = (Lambda < 0) @ w
    
    # Replace the average with the product
    w_product = w_u * w_v
    
    # Compute conductivity
    sigma = sigma_0 + gamma * (w_product)
    
    # Update sigma matrix
    K = diags(sigma)

    return K, sigma
""" parameters """

# paths for the images and the gif
path_images = "03_images\\"
manage_directory(path_images)
path_gif = "03_simulation.gif"

# colors of the plot (for the cluster)
colors = ListedColormap(["#f5c2ff", "#970B7B"])

n = 100 # grid size
length = 1 # egde length
sigma_0 = 0.01 # basic dry system conductivity
alpha = 0.5 # thermal diffusivity
k = 1.0 # cooling constant
k_w = 8.0 # drying constant
gamma = 40.0 # humidity-conductivity relation constant
I_target = 25 # to rescale Psi in the direct current case
threshold = 0.000999 # to narrow down the new cluster members
dt = 0.1
V_max_factor = 20.0  
base_V = 0           


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
Lambda = incidence_matrix(n)
L = length * identity(edges)
L_inverse = 1/length * identity(edges) # L is diagonal, so its inverse is trivial
T = np.zeros(nodes)
T[hotspots] = 1 # for burned nodes
w = np.ones(nodes) # it starts moisturized
K, sigma = compute_conductivity(Lambda, w, sigma_0, gamma)

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
        
        
        Psi = compute_potential(Lambda, K, L, L_inverse, S, coldspots, hotspots)
        global base_V 
        

        if alternating_current != 1:
            I_edges = K @ L_inverse @ Lambda @ Psi
            
           
            I_raw = np.sum(abs((Lambda.T @ I_edges)[coldspots]))
            if I_raw < 1e-20: I_raw = 1e-20 
            
        
            scale = I_target / I_raw
            
        
            if i == 0: base_V = scale
            
        
            if scale > base_V * V_max_factor:
                scale = base_V * V_max_factor 
                
      
            Psi = Psi * scale

        
        
        if i % 10 == 0:
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

        if i % 10 == 0:
        # plot - temperature
            cells[1].set_title(f'$T$, iteration #{i}, \n $k=${k}, $\\alpha=${alpha}')
            cell_1 = cells[1].imshow(T.reshape((n, n)), interpolation='nearest', cmap="PuRd")
            figure.colorbar(cell_1, ax=cells[1])

        # update humidity
        w = compute_humidity(w, T, k_w, dt)

        #np.set_printoptions(threshold=np.inf)
        #print(colored(w[np.where(cluster != 0)], "blue")) # TEMP


        if i % 10 == 0:
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
