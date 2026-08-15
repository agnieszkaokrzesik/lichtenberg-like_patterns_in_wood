code_version = '2026.07.20'

# introduced stagnation buffer to save active evolution of the system (for proper final plots)

# modified stagnant simulation cases to plot the last active frame established from the buffer
# added hotspots and coldspots to logs, moved to actual model time (instead of iterations) in statistics plots
# caution: cross-sections do not make sense with diagonally situated electrodes
# caution #2: versions from 2026.06.22 were not tested for current_scaling=True and may not work properly

prints = 0

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.sparse import csr_matrix, identity, diags
from scipy.sparse.linalg import spsolve
import random
import glob
import imageio.v2 as imageio
from termcolor import colored

# """ creates or clears a directory """
# def manage_directory(path):
#     # create a folder if it doesn't exist
#     if not os.path.exists(path): os.makedirs(path)
#     # clear the folder if it exists
#     if os.path.exists(path):
#         files = glob.glob(path + '*.png')
#         for f in files: os.remove(f)

""" returns incidence matrix for a (x, y) grid """
def incidence_matrix(x, y):

    nodes = x * y
    edges = (x - 1) * y + x * (y - 1)

    # arrays to store data needed to construct a sparse matrix
    rows = []
    cols = []
    values = []

    # helper function to save data for the sparse matrix
    def save_for_sparse(row, col, value):
        rows.append(row)
        cols.append(col)
        values.append(value)

    # take a node (0-based indexing)
    for v in range(nodes):
        # check its neighboring edges
        # left neighbor
        if (v % x != 0):
            left_edge = v - v//x - 1
            # save the coordinates and the value for the sparse matrix
            save_for_sparse(left_edge, v, 1)
        # right neighbor
        if (v % x != x - 1):
            right_edge =  v - v//x
            save_for_sparse(right_edge, v, -1) # minus marks the direction
        # top neighbor
        if v >= x:
            top_edge = (x - 1)*y + v - x
            save_for_sparse(top_edge, v, 1)
        # bottom neighbor
        if v < x*(y - 1):
            bottom_edge = (x - 1)*y + v
            save_for_sparse(bottom_edge, v, -1) # minus marks the direction
    
    incidence_matrix = csr_matrix((values, (rows, cols)), shape=(edges, nodes))

    return incidence_matrix

""" the main simulation function """
def simulation(
    run_timestamp, path_run, seed, grid_size, # setup
    V, J, U, A, W, chi, dt, eta=2, # parameters
    adaptive_dt=True, current_scaling=False, power_based_threshold=False, # physics-related options
    cap_from_logs=False, path_uncapped_logs=None, psi_max_fraction=0.5, # option: adjust cap based on previous run
    plot_interval=50, vector_grid=11, resistance_plot=True, raw_cluster_plot=True, animation=True, # visualization parameters
    cross_section=False, cross_section_index=None, cross_section_axis='y'): # cross-section options

    """ creates helper matrices to take care of mirror BCs """
    def build_ghost_operators(x, y, Lambda):

        nodes = x * y
        edges = (x - 1) * y + x * (y - 1)
        horizontal_edges = (x - 1) * y # horizontal edges are indexed first

        rows = np.arange(y)
        cols = np.arange(x)

        # boundary nodes
        left_nodes = rows * x
        right_nodes = left_nodes + (x - 1)
        top_nodes = cols
        bottom_nodes = top_nodes + x * (y - 1)

        # inward edges associated with those boundary nodes
        # horizontal
        left_edges = rows * (x - 1)
        right_edges = left_edges + (x - 2)
        # vertical
        top_edges = cols + horizontal_edges
        bottom_edges = top_edges + x * (y - 2)

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
        R_abs = csr_matrix((np.ones_like(mirror_nodes, dtype=float), (mirror_nodes, mirror_edges)), shape=(nodes, edges))

        # R_sgn: same selector, but with incidence signs for the Laplacian correction
        signs = np.array([Lambda[e, v] for v, e in zip(mirror_nodes, mirror_edges)], dtype=float)
        R_sgn = csr_matrix((signs, (mirror_nodes, mirror_edges)), shape=(nodes, edges))

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

    """ helper that computes total _node-based_ current leaving the cluster via coldspots """
    def compute_node_current_outflow(Lambda, Lambda_T, psi, K, L_inverse, coldspots):
        # currents along all edges
        delta_psi = Lambda @ psi
        I_edges = K @ L_inverse @ delta_psi
        # current in all nodes
        I_nodes = Lambda_T @ I_edges
        # total current leaving the cluster (via coldspots)
        I_out = np.sum(np.abs(I_nodes[coldspots]))
    
        return I_out

    """ scales potential using the total current leaving the cluster,
        and caps the potential to the maximum allowed value U if |psi_max| > U """
    def scale_and_cap_potential(K, L_inverse, Lambda, psi, coldspots, J, U, potential_cap):

        # compute total current leaving the cluster (via coldspots)
        I_out = compute_node_current_outflow(Lambda, Lambda_T, psi, K, L_inverse, coldspots)

        # avoid division by 0
        if I_out < 1e-20: I_out = 1e-20 
        
        # potential scaling
        scale = J / I_out
        psi *= scale

        # potential cap
        abs_psi = np.abs(psi)
        psi_max = np.max(abs_psi)
        if psi_max > U:
            # rescale the potential
            psi *= U / psi_max
            potential_cap = True
            if prints==1: print(colored("potential cap: ON", "yellow"))
        else:
            potential_cap = False
            if prints==1: print(colored("potential cap: OFF", "red"))

        uncapped_psi_max = psi_max

        if prints==1: print("psi_max", psi_max)

        return psi, uncapped_psi_max, potential_cap

    """ computes sums of horizontal and vertical currents
        in the main lattice divided into blocks """ # for current vector field plotting
    def compute_block_currents(n, m, I_edges):

        # size of a single lattice block
        block_size = (n - 1) // m

        # initialization for storing sum of x and y components for all the m x m blocks
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
    def compute_temperature(Lambda, Lambda_T, L_inverse, R_sgn, R_abs, psi, sigma, A, nodes):

        # diffusion term
        # J = A (Lambda_T L^-1 Lambda) T
        laplacian = Lambda_T @ L_inverse @ Lambda
        ghost_correction = R_sgn @ L_inverse @ Lambda
        laplacian_ghost = laplacian + ghost_correction

        # heat term
        # Q = 1/2 Lambda_T sigma (L^-1 Lambda psi)^2
        delta_psi = Lambda @ psi
        # heat along all edges
        Q_edges = sigma * (L_inverse @ delta_psi)**2
        # heat in the nodes, with 0.5 factor as each edge heats two nodes
        Q_nodes = 0.5 * np.abs(Lambda_T) @ Q_edges
        Q_ghost = 0.5 * R_abs @ Q_edges
        Q = Q_nodes + Q_ghost

        # T = [A (Lambda_T L^-1 Lambda) + 1]^-1 Q
        term = A * laplacian_ghost + identity(nodes)
        T = spsolve(term, Q)

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
    
    """ finds threshold-satisfying cluster edges
        * finds the edges that neighbor the cluster
        * narrows them down to those satisfying the dimensionless threshold:
          I^2 / sigma > 1 (or |I| > 1) """
    def find_eligible_edges(Lambda, cluster, psi, L_inverse, sigma):

        """ find the cluster's neighbors """
        # (term has negative values for edges that neighbor the cluster)
        term = Lambda @ cluster
        # get neighboring edges' indices
        neighboring_edges = np.where(term != 0)[0] # np.where returns a tuple

        """ compute currents """
        # compute delta_psi along the edges
        delta_psi = Lambda @ psi
        # currents along the edges
        I_edges = K @ L_inverse @ delta_psi

        """ narrow down to neighbors where I^2 / sigma > 1 (or |I| > 1) """ # 1 results from the dimensionless approach

        if power_based_threshold:
            P_edges = I_edges[neighboring_edges]**2 / sigma[neighboring_edges]
            threshold_parameter = P_edges
            condition = threshold_parameter > 1
        else:
            # current-based threshold
            abs_I = np.abs(I_edges[neighboring_edges])
            threshold_parameter = abs_I
            condition = threshold_parameter > 1

        eligible_edges = neighboring_edges[condition]

        if prints == 1: print(max(threshold_parameter))

        return eligible_edges, neighboring_edges, I_edges, np.max(threshold_parameter)
    
    """ computes power-based activation rates for eligible edges: 
        r = chi |I|^eta / sigma """
    def compute_activation_rates(I_edges, eligible_edges, sigma):
        # probability rates: r = chi |I|^eta / sigma (eta=2 by default)
        rates = chi * (np.abs(I_edges[eligible_edges])**eta / sigma[eligible_edges])

        return rates

    """ selects cluster's new members with probabilities p(dt) = r dt = chi |I|^eta / sigma dt """
    def find_new_members(rates, dt, eligible_edges, Lambda, cluster): 

        """ draw new member(s) using power-based activation rates """
        # p(dt) = r dt
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

        return new_nodes

    """ creates a three-panel plot visualizing cluster, temperature and humidity """
    def plot_system_state(
        run_timestamp,
        seed, x, y, i, cluster, I_x_sums, I_y_sums, T, w,
        V, J, U, A, W, chi, dt, current_scaling, adaptive_dt,
        potential_cap, t_total, final_frame=False, current_vectors=True,
        cluster_colormap=ListedColormap(["#f5c2ff", "#970B7B"]),
        T_colormap='PuRd', w_colormap='Purples'):

        vectors_caption = '' if current_vectors else '_clear'

        # plot initialization
        figure, cells = plt.subplots(3, 1, figsize=(5, 12), dpi=200)# dpi=300 - for better-quality representative plots
        cap_label = 'ON' if potential_cap else 'OFF'
        current_caption = fr'$V = {V}$, $I = $const., $J = {J}$, $U = {U:.2f}$, cap: {cap_label}' if current_scaling else fr'$V = {V}$, $\psi = $const.'
        threshold_label = 'power' if power_based_threshold else 'current'
        threshold_caption = f'{threshold_label}-based eligible edges threshold'
        final_caption = f'final frame, run {run_timestamp}' if final_frame else ''
        dt_caption = '(adaptive)' if adaptive_dt else ''
        figure.suptitle(
            f'iteration #{i}, seed {seed}, \n'
            f'{current_caption}, \n'
            f'{threshold_caption}, \n'
            f'$\\eta = {eta}$, total $t = {t_total:.5f}$ {dt_caption} \n'
            f'{final_caption}'
        )

        # plot - cluster
        cells[0].set_title(fr'cluster and currents')
        cells[0].imshow(cluster.reshape((y, x)), interpolation='nearest', cmap=cluster_colormap)

        if current_vectors:
            # plot - current vector field
            # block grid
            block_size = (n - 1) / m
            start = block_size / 2 # block-centered
            end = (n - 1) - start
            # coordinates of block centers
            x_points = np.linspace(start, end, m)
            y_points = np.linspace(start, end, m)
            X, Y = np.meshgrid(x_points, y_points)
            # vectors (by default starting at their _tail_ at (X, Y))
            cells[0].quiver(X, Y, -I_x_sums, I_y_sums, color='white')

        # plot - temperature
        cells[1].set_title(f'temperature $T$, \n $A={A}$')
        cell_1 = cells[1].imshow(T.reshape((y, x)), interpolation='nearest', cmap=T_colormap, vmin=0, vmax=np.max(T)) # fixed scale to suppress initial-frame noise
        figure.colorbar(cell_1, ax=cells[1])

        # plot - humidity
        cells[2].set_title(f'humidity $w$, \n $\\chi = {chi}$, $W={W}$')
        cell_2 = cells[2].imshow(w.reshape((y, x)), interpolation='nearest', cmap=w_colormap, vmin=0, vmax=1) # vmin, vmax - fixed scale
        figure.colorbar(cell_2, ax=cells[2])

        figure.tight_layout() # must be after plotting
        frame_timestamp = f'_{run_timestamp}' if final_frame else '' # timestamp only for the last frame
        plt.savefig(path_run + f'frame_{i:05d}{frame_timestamp}{vectors_caption}.png')
        plt.close()

    """ creates a three-panel plot visualizing vertical or horizontal cross-sections 
        of potential, temperature, and humidity (at a given cross_section_index) """
    def plot_cross_section(psi, w, T, i, x, y, cross_section_index, cross_section_axis):

        if cross_section_index == None: cross_section_index = x//2 if cross_section_axis=='x' else y//2 # default index value
        
        figure, cells = plt.subplots(3, 1, figsize=(5, 12)) 
        figure.subplots_adjust(hspace=0.2, top=0.93, bottom=0.05) # control subplots spacing
        
        # get labels and slices, depending on the orientation
        if cross_section_axis == 'x':
            label = 'vertical'
            axis = 'x'
            psi_section = psi.reshape((y, x))[:, cross_section_index]
            T_section = T.reshape((y, x))[:, cross_section_index]
            w_section = w.reshape((y, x))[:, cross_section_index]
        elif cross_section_axis == 'y':
            label = 'horizontal'
            axis = 'y'
            psi_section = psi.reshape((y, x))[cross_section_index, :]
            T_section = T.reshape((y, x))[cross_section_index, :]
            w_section = w.reshape((y, x))[cross_section_index, :]

        # plot caption
        figure.suptitle(fr'{label} cross-section at ${axis}={cross_section_index}$, iteration #{i}')

        # plot - potential
        cells[0].set_title(fr'$\psi$')
        cells[0].plot(psi_section, color="#F36CD8")

        # plot - temperature
        cells[1].set_title(fr'$T$')
        cells[1].plot(T_section, color="#FF0066")

        # plot - humidity
        cells[2].set_title(fr'$w$')
        cells[2].plot(w_section, color="#6724B3")

        plt.savefig(path_run + f'cross_section_{label}_{i:05d}.png')
        plt.close()

    """ buffer parameters of a stagnant system 
        (needed to plot buffered frames once new members are added again) """
    def buffer_stagnant_state(i, cluster_hot, cluster_cold,
                            I_x_sums, I_y_sums, psi,
                            T, w, dt, t_total):
        
        i_values.append(i)
        cluster_hot_values.append(cluster_hot)
        cluster_cold_values.append(cluster_cold)
        I_x_sums_values.append(I_x_sums)
        I_y_sums_values.append(I_y_sums)
        psi_values.append(psi)
        T_values.append(T)
        w_values.append(w)
        dt_values.append(dt)
        t_total_values.append(t_total)

    """ creates a 4 x 3 grid of plots depicting various system quantities over time """
    def plot_time_evolution(run_timestamp, U, path_logs):

        figure, cells = plt.subplots(5, 3, figsize=(12, 15), dpi=200)
        figure.suptitle(f'run {run_timestamp}')

        # extract system quantities from the log file
        log_file.flush() # enforce buffered data to be saved
        (uncapped_psi_max, psi_max, _, I_out, R_out, # (_ stands for the potential_cap flag)
        threshold_parameter_max_hot, neighboring_edges_hot, eligible_edges_hot, new_members_hot, 
        threshold_parameter_max_cold, neighboring_edges_cold, eligible_edges_cold, new_members_cold,
        T_mean, T_min, T_max, w_mean, w_min, w_max, dt) = np.genfromtxt(path_logs, dtype=float, skip_header=9, usecols=range(1, 21)).T

        t = np.cumsum(dt) # cumulative time

        # potential, current and resistance:
        # U and psi_max
        cells[0,0].set_title(r'$\psi_{\mathrm{max}}(t)$ and $U(t)$')
        cells[0,0].axhline(y=U, color="black", label=r'$U$')
        cells[0,0].plot(t, uncapped_psi_max, color="#970B7B", label=r'uncapped $\psi_{\mathrm{max}}$')
        cells[0,0].plot(t, psi_max, color="#F36CD8", label=r'$\psi_{\mathrm{max}}$')
        cells[0,0].legend()
        # total I_out
        cells[0,1].set_title(r'total $I_{\mathrm{out}}(t)$')
        cells[0,1].plot(t, I_out, color="#F36CD8")
        # total R_out
        cells[0,2].set_title(r'total $R_{\mathrm{out}}(t)$')
        cells[0,2].plot(t, R_out, color="#F36CD8")

        # new members - hot cluster:
        # maximum rate
        cells[1,0].set_title(r'max eligible edges threshold' + '\n (hot cluster)') # (first 600 iterations)
        cells[1,0].plot(t, threshold_parameter_max_hot, color="#970B7B") # threshold_parameter_max[:600]
        # neighbors edges and conducting edges
        cells[1,1].set_title(r'neighbors edges and eligible edges counts' + '\n (hot cluster)')
        cells[1,1].plot(t, neighboring_edges_hot, color="#F36CD8", label='neighbors edges')
        cells[1,1].plot(t, eligible_edges_hot, color="#970B7B", label='eligible edges')
        cells[1,1].legend()
        # new members
        cells[1,2].set_title(r'new members counts' + '\n (hot cluster)')
        cells[1,2].vlines(t, ymin=0, ymax=new_members_hot, colors="#970B7B")

        # new members - cold cluster:
        # maximum rate
        cells[2,0].set_title(r'max eligible edges threshold' + '\n (cold cluster)') # (first 600 iterations)
        cells[2,0].plot(t, threshold_parameter_max_cold, color="#2E0B82") # threshold_parameter_max[:600]
        # neighbors edges and conducting edges
        cells[2,1].set_title(r'neighbors edges and eligible edges counts' + '\n (cold cluster)')
        cells[2,1].plot(t, neighboring_edges_cold, color="#A06CE0", label='neighbors edges')
        cells[2,1].plot(t, eligible_edges_cold, color="#2E0B82",label='eligible edges')
        cells[2,1].legend()
        # new members
        cells[2,2].set_title(r'new members counts' + '\n (cold cluster)')
        cells[2,2].vlines(t, ymin=0, ymax=new_members_cold, color="#2E0B82")

        # temperature:
        # T_mean
        cells[3,0].set_title(r'$T_{\mathrm{avg}}(t)$')
        cells[3,0].plot(t, T_mean, color="#FF0066")
        # T_min
        cells[3,1].set_title(r'$T_{\mathrm{min}}(t)$')
        cells[3,1].plot(t, T_min, color="#FF0066")
        # T_max
        cells[3,2].set_title(r'$T_{\mathrm{max}}(t)$')
        cells[3,2].plot(t, T_max, color="#FF0066")

        # humidity:
        # w_mean
        cells[4,0].set_title(r'$w_{\mathrm{avg}}(t)$')
        cells[4,0].plot(t, w_mean, color="#A06CE0")
        # w_min
        cells[4,1].set_title(r'$w_{\mathrm{min}}(t)$')
        cells[4,1].plot(t, w_min, color="#A06CE0")
        # w_max
        cells[4,2].set_title(r'$w_{\mathrm{max}}(t)$')
        cells[4,2].plot(t, w_max, color="#A06CE0")

        figure.tight_layout()
        plt.savefig(path_run + f'time_evolution_{run_timestamp}.png')
        plt.close()

    """ creates an individual plot depicting system's resistance over time 
        (for easier further visualizations) """
    def plot_resistance(run_timestamp,  path_run, log_file, path_logs):
        
        plt.plot(figsize=(12, 12))
        plt.title(r'total $R_{\mathrm{out}}(t)$, ' + f'run {run_timestamp}')

        log_file.flush() # force log file update
        dt = np.genfromtxt(path_logs, dtype=float, skip_header=9, usecols=20).T
        R_out = np.genfromtxt(path_logs, dtype=float, skip_header=9, usecols=5).T

        t = np.cumsum(dt) # cumulative time

        plt.plot(t, R_out, color="#F36CD8")
        plt.tight_layout()
        plt.savefig(path_run + f'time_evolution_resistance_{run_timestamp}.png')
        plt.close()

    """ plots the final frame of the system, its cross-section, and the entire time evolution """
    def create_final_plots(
            run_timestamp, seed,
            x, y, i, cluster, I_x_sums, I_y_sums, psi, T, w,
            J, U, A, W, chi, dt, current_scaling, adaptive_dt, # parameters
            potential_cap, t_total, # flags
            cross_section_index, cross_section_axis): # cross-section plot

        # visualise the cluster, temperature and humidity (last frame)
        plot_system_state(
            run_timestamp,
            seed, x, y, i, cluster, I_x_sums, I_y_sums, T, w,
            V, J, U, A, W, chi, dt, current_scaling, adaptive_dt,
            potential_cap, t_total, final_frame=True)
        
        # version without current vector field
        plot_system_state(
            run_timestamp,
            seed, x, y, i, cluster, I_x_sums, I_y_sums, T, w,
            V, J, U, A, W, chi, dt, current_scaling, adaptive_dt,
            potential_cap, t_total, final_frame=True, current_vectors=False)
        
        if cross_section:
            # plot the cross-section of potential, temperature, and humidity
            plot_cross_section(psi, w, T, i, x, y, cross_section_index, cross_section_axis)

        # depict various system quantities over time
        plot_time_evolution(run_timestamp, U, path_logs)

        if resistance_plot:
            # creates an individual plot depicting system's resistance over time 
            # (for easier further visualizations)
            plot_resistance(run_timestamp,  path_run, log_file, path_logs)

    """ create a black-and-white 8-bit image of the cluster
        (for the sake of ImageJ post-processing) """
    def create_raw_plot(cluster_combined, path_run, run_timestamp):
        
        raw_plot = 255 * (cluster_combined - np.min(cluster_combined)) / (np.max(cluster_combined) - np.min(cluster_combined)) # normalize to 0-255 grayscale range
        raw_plot = 255 - raw_plot # black and white swap
        raw_plot = raw_plot.astype(np.uint8) # 8-bit format

        plt.imsave(path_run + f'raw_cluster_{run_timestamp}.png', raw_plot.reshape((y, x)), cmap="gray", vmin=0, vmax=255)
        plt.close()

    """ makes the animation depicting the system's evolution """
    def create_animation(path_run, run_timestamp):

        path_gif = path_run + f"/simulation_{run_timestamp}.gif"
        filenames = [f for f in glob.glob(path_run + 'frame*.png') if not f.endswith('_clear.png')]
        with imageio.get_writer(path_gif, mode='I', fps=4, dpi=200) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        print("the animation is ready.")

    """ creates final plots, animation and files """
    def finalize_simulation(
            run_timestamp, seed,
            x, y, i, cluster_hot, cluster_cold, I_x_sums, I_y_sums, psi, T, w,
            J, U, A, W, chi, dt, current_scaling, adaptive_dt,
            potential_cap, t_total,
            cross_section_index, cross_section_axis,
            path_run, path_cluster, path_potential, path_termination,
            stagnation_limit, cause):
        
        if cause == 'stagnation':
            final_text = f"no new members in {stagnation_limit} consecutive iterations, termination."
        else:
            final_text = f"the clusters have merged."

        print(final_text)

        cluster_combined = cluster_hot + cluster_cold
            
        # plot the final frame of the system, its cross-section, and the entire time evolution
        create_final_plots(
            run_timestamp, seed,
            x, y, i, cluster_combined, I_x_sums, I_y_sums, psi, T, w,
            J, U, A, W, chi, dt, current_scaling, adaptive_dt, # parameters
            potential_cap, t_total, # flags
            cross_section_index, cross_section_axis) # cross-section plot

        if raw_cluster_plot:
            # create a black-and-white 8-bit image of the cluster (for the sake of ImageJ post-processing)
            create_raw_plot(cluster_combined, path_run, run_timestamp)

        if animation:
            # make the animation depicting the system's evolution
            create_animation(path_run, run_timestamp) 

        # save final cluster
        np.savetxt(path_cluster, cluster_combined.reshape(1, -1), fmt="%d")

        # save final potential
        np.savetxt(path_potential, psi.reshape(1, -1))

        # save loop termination cause
        np.savetxt(path_termination, [cause], fmt="%s")


    print("the simulation has started...")

    # use potential cap from previous logs
    if cap_from_logs:

        # read both uncapped_psi_max and psi_max for sanity check
        (psi_log, psi_log_check) = np.genfromtxt(path_uncapped_logs, dtype=float, skip_header=9, usecols=(1, 2)).T
                                        
        if any(psi_log != psi_log_check):
            # sanity check
            raise ValueError("uncapped_psi_max and psi_max don't match. the provided logs are _not from an uncapped run_.")
        else:
            # set U to a fraction of psi_max
            abs_psi = np.abs(psi)
            U = np.max(abs_psi) * psi_max_fraction
            if prints==1: print("new cap", U)

    if cross_section_axis not in ('x', 'y'):
        raise ValueError("cross_section_axis must be either 'x' or 'y'.")

    # shorthand variables
    x = grid_size
    y = grid_size
    n = x
    m = vector_grid

    # randomness control
    random.seed(seed)
    np.random.seed(seed)

    """ initialization """

    t_total = 0
    potential_cap = False # potential cutoff flag

    # initialize dt if adaptive_dt is on
    dt = 1e-10 if adaptive_dt else dt

    nodes = x * y
    edges = (x - 1) * y + x * (y - 1)

    # BCs: electrode positions
    margin = x//5
    hotspots = [x*y*1//5 + margin] # psi > 0
    coldspots = [x*y*4//5 + x - margin] # psi = 0
    if len(hotspots) != 1 or len(coldspots) != 1:
        raise ValueError("'coldspots' and 'hotspots' must each contain only one element.")

    # source term, which marks the hotspots
    S = np.zeros(nodes)
    S[hotspots] = V

    # vector containing _values_ of BCs 
    boundary_conditions = np.zeros(nodes)
    boundary_conditions[hotspots] = 1
    
    # initialise the vectors and matrices
    Lambda = incidence_matrix(x, y)
    Lambda_T = Lambda.T
    R_abs, R_sgn = build_ghost_operators(x, y, Lambda)
    l = 1 # #TEMP reminder that length may become non-constant
    L = l * identity(edges)
    L_inverse = 1/l * identity(edges) # L is diagonal, so its inverse is trivial
    T = np.zeros(nodes)
    T[hotspots] = 1 # for burned nodes
    w = np.ones(nodes) # it starts moisturized
    K, sigma = compute_conductivity(Lambda, w, W)

    # vectors to mark nodes where carbon is
    cluster_hot = np.zeros(nodes) # positive electrode
    cluster_cold = np.zeros(nodes) # negative electrode
    # initialise the clusters according to BCs
    cluster_hot[hotspots] = 1
    cluster_cold[coldspots] = 1

    # create log file
    path_logs = path_run + f"/_logs_{run_timestamp}.txt"
    log_file = open(path_logs, "w")
    # header
    log_file.write(f'run timestamp: {run_timestamp} \n')
    log_file.write(f'code version: {code_version} \n')
    log_file.write(f'seed: {seed} \n')
    log_file.write(f'grid_size_x={x}, grid_size_y={y} \n')
    log_file.write(f'hotspot={hotspots[0]}, coldspot={coldspots[0]} \n')
    log_file.write(f'V={V}, J={J}, U={U}, A={A}, W={W}, chi={chi}, eta={eta} \n')
    log_file.write(f'adaptive_dt={adaptive_dt}, current_scaling={current_scaling}, power_based_threshold={power_based_threshold} \n')
    log_file.write(f'\n')
    log_file.write(f'iteration\t' +
                   f'uncapped_psi_max\tpsi_max\tpotential_cap\ttotal_I_out\ttotal_R_out\t' +
                   f'threshold_parameter_max_hot\tneighboring_edges_hot\teligible_edges_hot\tnew_members_hot\t' +
                   f'threshold_parameter_max_cold\tneighboring_edges_cold\teligible_edges_cold\tnew_members_cold\t' +
                   f'T_mean\tT_min\tT_max\t' +
                   f'w_mean\tw_min\tw_max\tdt \n')
    
    # path for the final cluster file
    path_cluster = path_run + f"/_final_cluster_{run_timestamp}.txt"

    # path for the final potential file
    path_potential = path_run + f"/_final_potential_{run_timestamp}.txt"

    # path for the loop termination cause
    path_termination = path_run + f"/_termination_cause_{run_timestamp}.txt"
    
    # new_members initialization in case no members are added before the first log entry
    new_members_hot = np.array([], dtype=int)
    new_members_cold = np.array([], dtype=int)

    """ the main loop """

    for i in range(0, 2*x*y):

        if i==0: plotting_active=True # True after cluster's growth, False after stagnation
        
        # compute the potential
        psi = compute_potential(Lambda, K, L_inverse, S, coldspots, hotspots)

        # current scaling case (direct current): rescale psi 
        if current_scaling:
            # scale the potential (using the total current leaving the cluster)
            # and cap the potential to the maximum allowed value U if |psi_max| > U
            # return the rescaled potential, uncapped psi_max and a cap flag
            psi, uncapped_psi_max, potential_cap = scale_and_cap_potential(K, L_inverse, Lambda, psi, coldspots, J, U, potential_cap)
        else:
            abs_psi = np.abs(psi)
            uncapped_psi_max = np.max(abs_psi)

        # compute total resistance of the system
        I_out = compute_node_current_outflow(Lambda, Lambda_T, psi, K, L_inverse, coldspots)
        psi_hot = psi[hotspots][0]
        psi_cold = psi[coldspots][0] # currently 0, but let's keep it explicit just in case
        R_out = np.abs(psi_hot - psi_cold) / I_out

        # currents computation for vector field plotting
        I_edges = K @ L_inverse @ Lambda @ psi
        # compute sums of x and y current components in m x m blocks
        I_x_sums, I_y_sums = compute_block_currents(n, m, I_edges)

        # compute the temperature
        T = compute_temperature(Lambda, Lambda_T, L_inverse, R_sgn, R_abs, psi, sigma, A, nodes)

        # update humidity
        w = compute_humidity(w, T, dt)

        # plot only if the cluster is active
        # (stagnant cluster's history is plotted only after it becomes active again)
        if plotting_active: 
            # visualise the cluster, temperature and humidity
            if i % plot_interval == 0:
                plot_system_state(
                    run_timestamp,
                    seed, x, y, i, cluster_hot + cluster_cold, I_x_sums, I_y_sums, T, w,
                    V, J, U, A, W, chi, dt, current_scaling, adaptive_dt,
                    potential_cap, t_total, final_frame=False)
                
                if cross_section:
                    # plot the cross-section of potential, temperature, and humidity
                    plot_cross_section(psi, w, T, i, x, y, cross_section_index, cross_section_axis)
            
            # save (possibly) last active state (for final plot's sake)
            last_active_state = (i, cluster_hot.copy(), cluster_cold.copy(),
                                    I_x_sums, I_y_sums, psi, T, w, dt, t_total)

        # finds threshold-satisfying cluster edges for both clusters
        (eligible_edges_hot, neighboring_edges_hot,
         I_edges_hot, threshold_parameter_max_hot) = find_eligible_edges(Lambda, cluster_hot, psi, L_inverse, sigma)   
        (eligible_edges_cold, neighboring_edges_cold,
         I_edges_cold, threshold_parameter_max_cold) = find_eligible_edges(Lambda, cluster_cold, psi, L_inverse, sigma)

        # if no edges met the threshold in both clusters
        if len(eligible_edges_hot) + len(eligible_edges_cold) == 0: 
            new_members_hot = np.array([], dtype=int)
            new_members_cold = np.array([], dtype=int)
            dt_log = dt # save dt from this iteration for the log file
            t_total += dt
        else:
            # compute the activation rates
            rates_hot = compute_activation_rates(I_edges_hot, eligible_edges_hot, sigma)
            rates_cold = compute_activation_rates(I_edges_cold, eligible_edges_cold, sigma)

            # find the clusters' new member indices
            new_members_hot = find_new_members(rates_hot, dt, eligible_edges_hot, Lambda, cluster_hot)
            new_members_cold = find_new_members(rates_cold, dt, eligible_edges_cold, Lambda, cluster_cold)

            # adaptive dt case
            if adaptive_dt:
                # scale dt based on maximum probability rate from both clusters
                max_rate = max(
                    np.max(rates_hot) if len(rates_hot) else 0, # for safety in case if only one cluster has eligible edges
                    np.max(rates_cold) if len(rates_cold) else 0
                )
                dt_new = 1 / (10 * max_rate)
            else:
                # keep dt as it was
                dt_new = dt
            
            t_total += dt # advance t_total by the current time step
            dt_log = dt # save dt from this iteration for the log file
            dt = dt_new # update dt for the next iteration
        
        log_line = (f'{i}\t' +
                    f'{uncapped_psi_max}\t{np.max(psi)}\t{potential_cap}\t{I_out}\t{R_out}\t' +
                    f'{threshold_parameter_max_hot}\t{len(neighboring_edges_hot)}\t{len(eligible_edges_hot)}\t{len(new_members_hot)}\t' +
                    f'{threshold_parameter_max_cold}\t{len(neighboring_edges_cold)}\t{len(eligible_edges_cold)}\t{len(new_members_cold)}\t' +
                    f'{np.mean(T)}\t{np.min(T)}\t{np.max(T)}\t' +
                    f'{np.mean(w)}\t{np.min(w)}\t{np.max(w)}\t' + f'{dt_log} \n')

        # update the log file only if the cluster is active
        # (stagnant cluster's logs are plotted only after it becomes active again)
        if plotting_active:
            log_file.write(log_line)
        else:
            stagnant_log_lines.append(log_line) # initialized in the first iteration (before plotting_active can become False)

        if prints==1: print(colored(f'new members (hot): {new_members_hot}', "blue"))
        if prints==1: print(colored(f'new members (cold): {new_members_cold}', "blue"))

        # update hotspots, coldspots, source term, boundary conditions and the cluster
        hotspots = np.append(hotspots, new_members_hot)
        coldspots = np.append(coldspots, new_members_cold)
        S[hotspots] = V
        boundary_conditions[new_members_hot] = psi[new_members_hot]
        boundary_conditions[new_members_cold] = psi[new_members_cold]
        cluster_hot[new_members_hot] = 1
        cluster_cold[new_members_cold] = 1

        # update conductivity
        K, sigma = compute_conductivity(Lambda, w, W)  

        # stagnation case: loop termination
        stagnation_limit = 200
        if i == 0:
            # variable initialization for the first iteration
            size = np.sum(cluster_hot + cluster_cold)
            size_previous = size
            stagnant = 0
            # arrays to store parameters of the system while it is stagnant
            (i_values, cluster_hot_values, cluster_cold_values,
            I_x_sums_values, I_y_sums_values, psi_values,
            T_values, w_values, dt_values, t_total_values) = ([], [], [], [], [], [], [], [], [], [])
            # array to store log file lines while the system stagnant
            stagnant_log_lines = []
        else:
            # denote continuous stagnation in cluster size
            size = np.sum(cluster_hot + cluster_cold)
            if size == size_previous:
                stagnant += 1
                plotting_active = False # flag to avoid immediate plots in the next iteration
                # buffer parameters of a stagnant system 
                # (needed to plot buffered frames once new members are added again)
                buffer_stagnant_state(i, cluster_hot, cluster_cold, I_x_sums, I_y_sums, psi, T, w, dt, t_total)
            else:
                # active cluster
                stagnant = 0
                plotting_active = True

                # save (possibly) last active state (for final plot's sake)
                last_active_state = (i, cluster_hot.copy(), cluster_cold.copy(),
                                     I_x_sums, I_y_sums, psi, T, w, dt, t_total)
    
                # plot all the previous stagnant frames
                for j in range(len(i_values)):

                    (i_plot, cluster_hot_plot, cluster_cold_plot,
                     I_x_sums_plot, I_y_sums_plot, psi_plot,
                     T_plot, w_plot, dt_plot, t_total_plot) = (i_values[j], cluster_hot_values[j], cluster_cold_values[j],
                                                               I_x_sums_values[j], I_y_sums_values[j], psi_values[j],
                                                               T_values[j], w_values[j], dt_values[j], t_total_values[j])

                    # visualise the cluster, temperature and humidity
                    if i_plot % plot_interval == 0:
                        #print('and now readt to plot all the previous frames')
                        plot_system_state(
                            run_timestamp,
                            seed, x, y, i_plot, cluster_hot_plot + cluster_cold_plot,
                            I_x_sums_plot, I_y_sums_plot, T_plot, w_plot,
                            V, J, U, A, W, chi, dt_plot, current_scaling, adaptive_dt,
                            potential_cap, t_total_plot, final_frame=False)
                        
                        if cross_section:
                            # plot the cross-section of potential, temperature, and humidity
                            plot_cross_section(psi_plot, w_plot, T_plot, i_plot, x, y, cross_section_index, cross_section_axis)
                
                    # update the log file
                    log_file.write(stagnant_log_lines[j])

                # clear the arrays
                (i_values, cluster_hot_values, cluster_cold_values,
                I_x_sums_values, I_y_sums_values, psi_values,
                T_values, w_values, dt_values, t_total_values) = ([], [], [], [], [], [], [], [], [], [])
                stagnant_log_lines = []

            size_previous = size

            # terminate the loop if no members were added since ... iterations
            if stagnant == stagnation_limit:

                # retrieve last active state
                (i, cluster_hot, cluster_cold,
                 I_x_sums, I_y_sums, psi, T, w, dt, t_total) = last_active_state 

                finalize_simulation(
                    run_timestamp, seed,
                    x, y, i, cluster_hot, cluster_cold, I_x_sums, I_y_sums, psi, T, w,
                    J, U, A, W, chi, dt, current_scaling, adaptive_dt,
                    potential_cap, t_total,
                    cross_section_index, cross_section_axis,
                    path_run, path_cluster, path_potential, path_termination,
                    stagnation_limit, cause='stagnation')

                break

        # terminate the loop as soon as both clusters merge
        if np.sum((np.abs(Lambda_T @ Lambda) @ cluster_hot) * cluster_cold) > 0:

            finalize_simulation(
                run_timestamp, seed,
                x, y, i, cluster_hot, cluster_cold, I_x_sums, I_y_sums, psi, T, w,
                J, U, A, W, chi, dt, current_scaling, adaptive_dt,
                potential_cap, t_total,
                cross_section_index, cross_section_axis,
                path_run, path_cluster, path_potential, path_termination,
                stagnation_limit, cause='merge')
            
            break
    
    log_file.close()

# USEFUL printing
# for i in range (0, n**2, n): print(sources[i:i+n])
# np.set_printoptions(threshold=np.inf)
