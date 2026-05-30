# 2026.05.27

# only functions left (execution moved to separate scripts)

prints = 0

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.sparse import csr_matrix, identity, diags
from scipy.sparse.linalg import spsolve
import random
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
    path_images,
    seed, n, coldspots, hotspots, # setup
    J, U, A, W, chi, dt, eta=2, # parameters
    adaptive_dt=True, current_scaling=True, # options
    crossection_index=None, crossection_axis='y', m=10): # visualization parameters

    """ returns incidence matrix for an n x n grid """
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

    """ """
    def build_ghost_operators(n, Lambda):
        nodes = n**2
        edges = 2 * n * (n - 1)
        base = n * (n - 1)

        rows = np.arange(n)
        cols = np.arange(n)

        # Boundary nodes
        left_nodes = rows * n
        right_nodes = rows * n + (n - 1)
        top_nodes = cols
        bottom_nodes = (n - 1) * n + cols

        # Inward normal edges associated with those boundary nodes
        # Horizontal edges
        left_edges = rows * (n - 1)
        right_edges = rows * (n - 1) + (n - 2)

        # Vertical edges
        top_edges = base + cols
        bottom_edges = base + (n - 2) * n + cols

        mirror_nodes = np.concatenate((
            left_nodes,
            right_nodes,
            top_nodes,
            bottom_nodes
        ))

        mirror_edges = np.concatenate((
            left_edges,
            right_edges,
            top_edges,
            bottom_edges
        ))

        # R_abs: adds mirrored heat contribution only to the correct boundary node
        R_abs = csr_matrix(
            (
                np.ones_like(mirror_nodes, dtype=float),
                (mirror_nodes, mirror_edges)
            ),
            shape=(nodes, edges)
        )

        # R_sgn: same selector, but with incidence signs for the Laplacian correction
        signs = np.array([
            Lambda[e, v] for v, e in zip(mirror_nodes, mirror_edges)
        ], dtype=float)

        R_sgn = csr_matrix(
            (
                signs,
                (mirror_nodes, mirror_edges)
            ),
            shape=(nodes, edges)
        )

        return R_abs, R_sgn

    """ returns cleared version of a matrix used to enforce BCs """
    def term_BCs_enforced(Lambda, K, L_inverse, coldspots, hotspots):

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

    """ computes potential by solving a linear system (with enforced BCs):
        grad(sigma grad psi) = 0 """ #TEMP: what if L!=const??
    def compute_potential(Lambda, K, L_inverse, S, coldspots, hotspots):
        # psi = (Lambda^T K L^-1 Lambda)^-1 S
        bracket = term_BCs_enforced(Lambda, K, L_inverse, coldspots, hotspots) 
        # solve for potentials
        psi = spsolve(bracket, S)

        return psi

    """ scales potential using the total current leaving the cluster,
        and caps the potential to the maximum allowed value U if |psi_max| > U """
    def scale_and_cap_potential(K, L_inverse, Lambda, psi, coldspots, J, U, potential_cap):

        # helper function that computes total current leaving the cluster via coldspots
        def compute_current_outflow(K, L_inverse, Lambda, psi, coldspots):
            # currents along all edges
            I_edges = K @ L_inverse @ Lambda @ psi
            # current in all nodes
            I_nodes = Lambda.T @ I_edges
            # total current leaving the cluster (via coldspots)
            I_out = np.sum(abs(I_nodes[coldspots]))

            return I_out

        # compute total current leaving the cluster (via coldspots)
        I_out = compute_current_outflow(K, L_inverse, Lambda, psi, coldspots)

        # avoid division by 0
        if I_out < 1e-20: I_out = 1e-20 
        
        # potential scaling
        scale = J / I_out
        psi *= scale

        # potential cap
        psi_max = np.max(np.abs(psi))
        if psi_max > U:
            # rescale the potential
            psi *= U / psi_max
            potential_cap = True
            if prints==1: print(colored("potential cap: ON", "yellow"))
        else:
            potential_cap = False
            if prints==1: print(colored("potential cap: OFF", "red"))

        if prints==1: print("psi_max", psi_max)

        return psi, potential_cap

    """ computes sums of horizontal and vertical currents
        in the main lattice divided into m x m blocks """ # for current vector field plotting
    def compute_block_currents(n, m, I_edges):

        # size of a single lattice block
        block_size = (n - 1) // m

        # initiation for storing sum of x and y components for all the m x m blocks
        I_x_sums = np.zeros((m, m))
        I_y_sums = np.zeros((m, m))

        # divide I_edges into the x and y parts
        horizontal = n * (n - 1) # first n*(n-1) edges are horizontal
        I_x = I_edges[:horizontal]
        I_y = I_edges[horizontal:]

        # reshape the arrays so they can be easily sliced later
        I_x = I_x.reshape((n, n - 1))
        I_y = I_y.reshape((n - 1, n))

        # loop over the blocks
        for i in range(m):
            for j in range(m):
                # edge-based coordinates
                row_start = i * block_size
                row_end = (i + 1) * block_size
                col_start = j * block_size
                col_end = (j + 1) * block_size

                # sum of x and y current components
                I_x_sums[i, j] = np.sum(I_x[row_start:row_end, col_start:col_end])
                I_y_sums[i, j] = np.sum(I_y[row_start:row_end, col_start:col_end])

        return I_x_sums, I_y_sums

    """ computes temperature by solving a linear system:
        A grad^2 T + T = sigma |grad psi|^2 """
    def compute_temperature(Lambda, L_inverse, R_sgn, R_abs, psi, sigma, A, nodes):

        # diffusion term
        # J = A (Lambda.T L^-1 Lambda) T
        laplacian = Lambda.T @ L_inverse @ Lambda
        ghost_correction = R_sgn @ L_inverse @ Lambda
        laplacian_ghost = laplacian + ghost_correction

        # heat term
        # Q = 1/2 Lambda.T sigma (L^-1 Lambda psi)^2
        delta_psi = Lambda @ psi
        # heat along all edges
        Q_edges = sigma * (L_inverse @ delta_psi)**2
        # heat in the nodes, with 0.5 factor as each edge heats two nodes
        #Q_nodes = 0.5 * abs(Lambda).T @ Q_edges
        q_eff = (0.5 * abs(Lambda).T + 0.5 * R_abs) @ Q_edges

        # # additional terms from mirror boundary conditions
        # mirror_laplacian = Lambda.T @ diags(B_e) @ Lambda @ diags(B_n)
        # mirror_Q = diags(B_n) @ np.abs(Lambda.T) @ diags(B_e) @ Q_edges

        # T = [A (Lambda.T L^-1 Lambda) + 1]^-1 Q
        #term = A * (laplacian + mirror_laplacian) + 1 * identity(nodes)
        #T = spsolve(term, Q_nodes + mirror_Q)
        term = A * laplacian_ghost + identity(nodes)
        T = spsolve(term, q_eff)

        return T

    """ computes humidity by integrating over time:
        dw/dt = -T w """
    def compute_humidity(w, T, dt):
        # function controlling drying - exponential decay
        def f(w, T):
            return w * T

        # integration
        w -= f(w, T) * np.heaviside(w, 0) * dt

        if prints==1: print("w_min", min(w))

        return np.maximum(w, 0) # to prevent negative w

    """ computes conductivity based on humidity:
        sigma = 1 + W w^2 """ #TEMP shorthand - w^2 actually stands for w_u w_v
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

    """ finds cluster's new member(s) out of its neighbors:
        * first narrows them down to those satisfying |I| > 1 (dimensionless threshold)
        * then selects them with probabilities P = chi I^eta / sigma dt """
    def find_new_members(Lambda, cluster, psi, K, L_inverse, chi, dt, t_total, eta, adaptive_dt): 

        """ find the cluster's neighbors """
        # (term has negative values for edges that neighbor the cluster)
        term = Lambda @ cluster
        # get neighboring edges' indices
        neighbors_edges = np.where(term != 0)[0] # np.where returns a tuple

        """ compute currents """
        # compute delta_psi along the edges
        delta_psi = Lambda @ psi
        # currents along the edges
        I_edges = K @ L_inverse @ delta_psi

        """ narrow down to neighbors where |I| > 1 """ # 1 results from the dimensionless approach
        I_abs = np.abs(I_edges[neighbors_edges])
        eligible_edges = neighbors_edges[I_abs > 1]
        # if no neighbors satisfy the condition
        if len(eligible_edges) == 0:
            # dt and total time handling
            dt_new = dt
            t_total += dt
            return np.array([], dtype=int), dt_new, t_total

        """ draw new member(s) using current-based activation rates """
        # probability rates: r = chi I^eta / sigma (eta=2 by default)
        rates = chi * (np.abs(I_edges[eligible_edges])**eta / sigma[eligible_edges])
        # P(dt) = r dt
        probabilities = rates * dt
        # draw a random threshold for each edge
        thresholds = np.random.rand(len(eligible_edges))
        # narrow down the edges again, using probabilities
        indices = np.where(thresholds < probabilities)[0]
        new_edges = eligible_edges[indices]

        """ find the node connected to the chosen edge """
        # find nodes that are connected by new_edges
        connected_nodes = Lambda[new_edges, :].nonzero()[1] # non-zero element from the chosen row; [1] to get only column indices
        # indices of the cluster elements
        cluster_nodes = np.where(cluster != 0)[0]
        # choose the ones of connected_nodes that do not belong to the cluster
        new_nodes = np.setdiff1d(connected_nodes, cluster_nodes)

        if prints==1: print("probab_max", max(probabilities))

        # adaptive dt case
        if adaptive_dt:
            # scale dt based on maximum probability rate
            max_rate = max(rates)
            dt_new = 1 / (10 * max_rate)
            t_total += dt_new
        else:
            # keep dt as it was
            dt_new = dt
            t_total += dt

        return new_nodes, dt_new, t_total

    """ creates a three-panel plot visualizing cluster, temperature and humidity """
    def plot_system_state(
        cluster_colormap, T_colormap, w_colormap,
        seed, n, i, cluster, I_x_sums, I_y_sums, T, w,
        J, U, A, W, chi, dt, current_scaling, adaptive_dt,
        potential_cap, t_total, final_frame_caption=False):

        # plot initiation
        figure, cells = plt.subplots(3, 1, figsize=(5, 12)) #, dpi=300) dpi=300 - for better-quality representative plots
        cap_value = 'ON' if potential_cap else 'OFF'
        current_caption = f'current scaling, $J = {J}$, $U = {U}$, cap: {cap_value}' if current_scaling else 'no current scaling'
        final_caption = f'final frame, total $t = {t_total:.5f}$' if final_frame_caption else ''
        dt_caption = '(adaptive)' if adaptive_dt else ''
        figure.suptitle(
            f'iteration #{i}, seed {seed}, \n {current_caption}, \n $\\chi = {chi}$, $\\eta = {eta}$, $dt = {dt:.3}$ {dt_caption} \n {final_caption}'
        )

        # plot - cluster
        cells[0].set_title(fr'cluster and currents')
        cells[0].imshow(cluster.reshape((n, n)), interpolation='nearest', cmap=cluster_colormap)

        # plot - current vector field
        # grid
        block_size = (n - 1) / m
        start = block_size / 2 # block-centered
        end = (n - 1) - start
        x = np.linspace(start, end, m)  
        y = np.linspace(start, end, m)
        X, Y = np.meshgrid(x, y)
        # vectors (by default starting at their _tail_ at (X, Y))
        cells[0].quiver(X, Y, -I_x_sums, I_y_sums, color='white')

        # plot - temperature
        cells[1].set_title(f'temperature $T$, \n $A={A}$')
        cell_1 = cells[1].imshow(T.reshape((n, n)), interpolation='nearest', cmap=T_colormap, vmin=0, vmax=max(T)) # fixed scale to suppress initial-frame noise
        figure.colorbar(cell_1, ax=cells[1])

        # plot - humidity
        cells[2].set_title(f'humidity $w$, \n $W={W}$')
        cell_2 = cells[2].imshow(w.reshape((n, n)), interpolation='nearest', cmap=w_colormap, vmin=0, vmax=1) # vmin, vmax - fixed scale
        figure.colorbar(cell_2, ax=cells[2])

        figure.tight_layout() # must be after plotting
        plt.savefig(path_images + f'simulation{i:05d}.png')
        plt.close()

    """ creates a three-panel plot visualizing vertical or horizontal cross-sections 
        of potential, temperature, and humidity (at a given crossection_index) """
    def plot_crosssection(psi, w, T, i, n, crossection_index, crossection_axis):

        if crossection_index == None: crossection_index = n//2 # default index value
        
        # plot initiation
        figure, cells = plt.subplots(3, 1, figsize=(5, 12)) 
        figure.subplots_adjust(hspace=0.2, top=0.93, bottom=0.05) # control subplots spacing
        
        # get labels and slices, depending on the orientation
        if crossection_axis == 'x':
            label = 'vertical'
            axis = 'x'
            psi_section = psi.reshape((n, n))[:, crossection_index]
            T_section = T.reshape((n, n))[:, crossection_index]
            w_section = w.reshape((n, n))[:, crossection_index]
        elif crossection_axis == 'y':
            label = 'horizontal'
            axis = 'y'
            psi_section = psi.reshape((n, n))[crossection_index, :]
            T_section = T.reshape((n, n))[crossection_index, :]
            w_section = w.reshape((n, n))[crossection_index, :]
        else:
            print('wrong crossection_axis value. only "x" and "y" are valid options.')

        # plot caption
        figure.suptitle(fr'{label} cross-section at ${axis}={crossection_index}$, iteration #{i}')

        # plot - potential
        cells[0].set_title(fr'$\psi$')
        cells[0].plot(psi_section, color="#F36CD8")

        # plot - temperature
        cells[1].set_title(fr'$T$')
        cells[1].plot(T_section, color="#FF0066")

        # plot - humidity
        cells[2].set_title(fr'$w$')
        cells[2].plot(w_section, color="#6724B3")

        plt.savefig(path_images + f'crosssection{i:05d}_{label}.png')
        plt.close()

    print("the simulation has started...")

    # colors of the plot (for the cluster)
    cluster_colormap = ListedColormap(["#f5c2ff", "#970B7B"])
    T_colormap = 'PuRd'
    w_colormap = 'Purples'

    # randomness control
    random.seed(seed)
    np.random.seed(seed)

    # how often a plot is saved (for the animation), adaptive to dt
    plot_interval = 100 #max(10, 1 // (dt * 10**3)) #TEMP0520

    """ initialisation """

    t_total = 0
    potential_cap = False # potential cutoff flag

    # initiate dt if adaptive_dt is on
    dt = 1e-10 if adaptive_dt else dt

    nodes = n**2
    edges = 2*n * (n - 1)

    # source term, which marks the hotspots
    S = np.zeros(nodes)
    S[hotspots] = 1

    # vector containing _values_ of BCs 
    boundary_conditions = np.zeros(nodes)
    boundary_conditions[hotspots] = 1
    
    # initialise the vectors and matrices
    Lambda = incidence_matrix(n)
    R_abs, R_sgn = build_ghost_operators(n, Lambda)
    l = 1 # #TEMP reminder that length may become non-constant
    L = l * identity(edges)
    L_inverse = 1/l * identity(edges) # L is diagonal, so its inverse is trivial
    T = np.zeros(nodes)
    T[hotspots] = 1 # for burned nodes
    w = np.ones(nodes) # it starts moisturized
    K, sigma = compute_conductivity(Lambda, w, W)

    # vector to mark nodes where carbon is
    cluster = np.zeros(nodes)
    # initialise the cluster according to BCs
    cluster[hotspots] = 1

    """ the main loop """

    for i in range(0, 2*n**2):
        
        # compute the potential
        psi = compute_potential(Lambda, K, L_inverse, S, coldspots, hotspots)

        # current scaling case (direct current): rescale psi 
        if current_scaling:
            
            # scale the potential (using the total current leaving the cluster)
            # and cap the potential to the maximum allowed value U if |psi_max| > U
            # return the rescaled potential and a cap flag
            psi, potential_cap = scale_and_cap_potential(K, L_inverse, Lambda, psi, coldspots, J, U, potential_cap)

        # currents computation for vector field plotting
        I_edges = K @ L_inverse @ Lambda @ psi
        # compute sums of x and y current components in m x m blocks
        I_x_sums, I_y_sums = compute_block_currents(n, m, I_edges)

        # compute the temperature
        T = compute_temperature(Lambda, L_inverse, R_sgn, R_abs, psi, sigma, A, nodes)

        # update humidity
        w = compute_humidity(w, T, dt)

        # visualise the cluster, temperature and humidity
        if i % plot_interval == 0:
            plot_system_state(
                cluster_colormap, T_colormap, w_colormap,
                seed, n, i, cluster, I_x_sums, I_y_sums, T, w,
                J, U, A, W, chi, dt, current_scaling, adaptive_dt,
                potential_cap, t_total, final_frame_caption=False)
            
            plot_crosssection(psi, w, T, i, n, crossection_index, crossection_axis)

        # find the cluster's new member index (and new dt - if adaptive_dt is on)
        new_members, dt, t_total = find_new_members(Lambda, cluster, psi, K, L_inverse, chi, dt, t_total, eta, adaptive_dt)

        if prints==1: print(colored(f'new members: {new_members}', "blue"))

        # update hotspots, source term, boundary conditions and the cluster
        hotspots = np.append(hotspots, new_members)
        S[hotspots] = 1
        boundary_conditions[new_members] = psi[new_members]
        cluster[new_members] = 1

        # update conductivity
        K, sigma = compute_conductivity(Lambda, w, W)

        # terminate the loop as soon as hotspots invade the last column
        last_column = [i + (n - 1) for i in range(0, n**2, n)]
        if np.any(cluster[last_column] != 0): 
            
            # visualise the cluster, temperature and humidity (last frame)
            plot_system_state(
                cluster_colormap, T_colormap, w_colormap,
                seed, n, i, cluster, I_x_sums, I_y_sums, T, w,
                J, U, A, W, chi, dt, current_scaling, adaptive_dt,
                potential_cap, t_total, final_frame_caption=True)
            
            plot_crosssection(psi, w, T, i, n, crossection_index, crossection_axis)
            
            break

# USEFUL printing
# for i in range (0, n**2, n): print(sources[i:i+n])
# np.set_printoptions(threshold=np.inf)
