# 2026.11.12-2

# corresponds to the above simulation_core version

import numpy as np
import os
import glob
from datetime import datetime
from termcolor import colored
from simulation_core import simulation

start_time = datetime.now()

cap_from_logs_value = False
uncapped_loop_timestamp = '20260611_001423'

# use previous test loop to obtain U value
if cap_from_logs_value:
    # path to the test loop
    path_uncapped_loop = f"output/sweep_run/loop_{uncapped_loop_timestamp}/"
    # paths to the runs inside the loop
    uncapped_runs = os.listdir(path_uncapped_loop)

    uncapped_logs = []

    # loop through the test runs
    for run in uncapped_runs:
        run_path = os.path.join(path_uncapped_loop, run)
        # loop through the files inside a run and find the .txt log file
        for file in os.listdir(run_path):
            if file.endswith(".txt"):
                log_file = os.path.join(run_path, file)
                uncapped_logs.append(log_file)

# path for the images folders
loop_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path_loop = f"output/sweep_run/loop_{loop_timestamp}/"
if not os.path.exists(path_loop): os.makedirs(path_loop)

parameter_rows = 'V'
parameter_cols = 'chi'

V_values = np.arange(2.0, 2.75, 0.25)
chi_values = np.arange(0.3, 0.6, 0.1)

# V_values = np.arange(1, 11, 1)
# chi_values = np.arange(1/3, 3+1/3, 1/3)

range_rows = V_values
range_cols = chi_values

# loop log file
path_loop_logs = path_loop + f"/loop_logs_{loop_timestamp}.txt"
loop_log_file = open(path_loop_logs, "w")
# parameter 1
loop_log_file.write(f'{parameter_rows}\n')
np.savetxt(loop_log_file, range_rows.reshape(1, -1), fmt='%g')
# parameter 2
loop_log_file.write(f'{parameter_cols}\n')
np.savetxt(loop_log_file, range_cols.reshape(1, -1), fmt='%g')
loop_log_file.close()

it = 0
for i in range_rows:
    for j in range_cols:

        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        
        progress = it / (len(range_cols) * len(range_rows))
        #print(f"iteration #{it}, chi = {chi}, J = {J}")
        #print(f"iteration #{it}, W = {W}, J = {J}")
        print(f"iteration #{it}, {str(parameter_rows)} = {i}, {str(parameter_cols)} = {j}")
        print(colored(f"{progress * 100:.1f}% of the loop", "blue"))

        """ initialisation """

        # path for the images
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_run = path_loop + f"simulation_{run_timestamp}/"
        if not os.path.exists(path_run): os.makedirs(path_run)

        # parameters

        seed = 42 # randomness control
        grid_size = 200

        V = i
        A = 1
        W = 150
        chi = j

        J = None
        U = None
        dt = None

        # run the simulation
        simulation(
            run_timestamp, path_run, seed, grid_size, # setup
            V, J, U, A, W, chi, dt, eta=2, # parameters
            adaptive_dt=True, current_scaling=False, power_based_threshold=False, # physics-related options
            cap_from_logs=False, path_uncapped_logs=None, psi_max_fraction=0.5, # option: adjust cap based on previous run
            plot_interval=50, vector_grid=11, # visualization parameters 
            pdf_states=False, i_pdf=(), # saving data for pdf plots
            resistance_plot=True, raw_cluster_plot=True, animation=True, # plot options
            cross_section=False, cross_section_index=None, cross_section_axis='y') # cross-section options

        it += 1
    
end_time = datetime.now()
execution_time = end_time - start_time

print(f"execution time: {execution_time}")
