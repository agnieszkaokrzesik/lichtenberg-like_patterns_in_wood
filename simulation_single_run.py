# 2026.10.02

# corresponds to the above simulation_core version

import numpy as np
import os
from datetime import datetime
from simulation_core import simulation

""" initialisation """

# paths for the images and the gif
start_time = datetime.now()
run_timestamp = start_time.strftime("%Y%m%d_%H%M%S")
path_run = f"simulation_single_run/simulation_{run_timestamp}/"
if not os.path.exists(path_run): os.makedirs(path_run)

# parameters

seed = 43 # randomness control
grid_size = 200

dt = 8e-6 # caution: overridden inside the simulation function if adaptive_dt (but that's ok) # UWAGA CZEMU TO NIE ROSNIE #TEMP 0609
V = 7
J = 100.0000001
U = np.inf # 200 # np.inf
A = 0.1
W = 100
chi = 1

# run the simulation
simulation(
    run_timestamp, path_run, seed, grid_size, # setup
    V, J, U, A, W, chi, dt, eta=2, # parameters
    adaptive_dt=True, current_scaling=False, power_based_threshold=False, # physics-related options
    cap_from_logs=False, path_uncapped_logs=None, psi_max_fraction=0.5, # option: adjust cap based on previous run
    plot_interval=50, vector_grid=11, resistance_plot=True, raw_cluster_plot=True, animation=True, # visualization parameters
    cross_section=True, cross_section_index=None, cross_section_axis='diagonal') # cross-section options

end_time = datetime.now()
execution_time = end_time - start_time

print(f"execution time: {execution_time}")