# 2026.06.07 v3

# corresponds to the above simulation_core version

import numpy as np
import os
import glob
from datetime import datetime
from termcolor import colored
from simulation_core import simulation

cap_from_logs_value = False
uncapped_loop_timestamp = '20260607_182450'

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

chi_values = range(1, 10)
J_values = 10.0**np.arange(2, 9)
W_values = 10.0**np.arange(-3, 9)

it = 0
for chi in chi_values:
#for W in W_values:
    for J in J_values: # caution: for J = 1, 10 the system got clogged
                       # because |I| > 1 (dimensionless threshold) was never satisfied
        
        progress = it / (len(W_values) * len(J_values))
        print(f"iteration #{it}, chi = {chi}, J = {J}")
        #print(f"iteration #{it}, W = {W}, J = {J}")
        print(colored(f"{progress * 100:.1f}% of the loop", "blue"))

        """ initialisation """

        # path for the images
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_run = path_loop + f"images_{run_timestamp}/"
        if not os.path.exists(path_run): os.makedirs(path_run)

        # parameters

        seed = 42 # randomness control
        n = 100 # grid size

        dt = 8e-6 # caution: overridden inside the simulation function if adaptive_dt (but that's ok)
        #J = 2500
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
            run_timestamp, path_run,
            seed, n, # setup
            J, U, A, W, chi, dt, eta=2)
            #cap_from_logs=cap_from_logs_value, path_uncapped_logs=uncapped_logs[it]) # parameters
    
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")

        it += 1
        