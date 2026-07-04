# 2026.07.02 v2

# corresponds to the above simulation_core version

# added execution time display

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
if cap_from_logs_value == True:
    # path to the test loop
    path_uncapped_loop = f"simulation_multiple_run/loop_{uncapped_loop_timestamp}/"
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
path_loop = f"simulation_multiple_run/loop_{loop_timestamp}/"
if not os.path.exists(path_loop): os.makedirs(path_loop)

V_values = np.arange(0.25, 2.75, 0.25)
chi_values = np.arange(0.1, 1.1, 0.1)

range_rows = V_values
range_cols = chi_values

it = 0
for V in range_rows:
    for chi in range_cols:
    #for J in range_cols:

        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        
        progress = it / (len(range_cols) * len(range_rows))
        #print(f"iteration #{it}, chi = {chi}, J = {J}")
        #print(f"iteration #{it}, W = {W}, J = {J}")
        print(f"iteration #{it}, chi = {chi}, V = {V}")
        print(colored(f"{progress * 100:.1f}% of the loop", "blue"))

        """ initialisation """

        # path for the images
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_run = path_loop + f"simulation_{run_timestamp}/"
        if not os.path.exists(path_run): os.makedirs(path_run)

        # parameters

        seed = 42 # randomness control
        n = 200 # grid size

        dt = 8e-6 # caution: overridden inside the simulation function if adaptive_dt (but that's ok)
        #V = 1
        J = 2500
        U = np.inf # or J / 100
        A = 1
        W = 100
        #chi = 4

        # BCs: store _indices_ of nodes where psi is 0 or > 0
        hotspots =  np.arange(0, n**2, n) # potential > 0
        coldspots = np.arange(n-1, n**2, n) # potential = 0

        #print(uncapped_logs[it])

        # run the simulation
        simulation(
            run_timestamp, path_run, seed, n, # setup
            V, J, U, A, W, chi, dt, eta=2,
            adaptive_dt=True, current_scaling=False, power_based_threshold=False,
            cap_from_logs=False, path_uncapped_logs=None, psi_max_fraction=None,
            cross_section=False, cros_section_axis='y', plot_interval=50)

        it += 1
    
end_time = datetime.now()
execution_time = end_time - start_time

print(f"execution time: {execution_time}")


