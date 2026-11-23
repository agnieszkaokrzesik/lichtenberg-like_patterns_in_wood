# 2026.11.18

# added seed loop and microsecond timestamp

# corresponds to the above simulation_core version

import numpy as np
import os
import glob
from datetime import datetime
from termcolor import colored
from simulation_core import simulation

for k in [42]: # seeds
    start_time = datetime.now()

    #loop_purpose = input("what's the purpose of this sweep? (enter = skip)\n")
    loop_purpose = f"(...) seed {k}"

    cap_from_logs_value = False
    uncapped_loop_timestamp = None

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

    V_values = np.arange(2.75, 5.125, 1/4)
    chi_values = np.arange(0.125, 1.125, 1/8)

    range_rows = V_values
    range_cols = chi_values

    # loop range files
    np.save(path_loop + f"/loop_range_rows_{parameter_rows}_{loop_timestamp}.npy", range_rows)
    np.save(path_loop + f"/loop_range_cols_{parameter_cols}_{loop_timestamp}.npy", range_cols)

    # loop message file
    with open(path_loop + f"loop_purpose_{loop_timestamp}.txt", "w", encoding="utf-8") as f:
        f.write(loop_purpose.strip() + '\n'
                + f'{parameter_rows}: {range_rows.reshape(1, -1)[0]}\n'
                + f'{parameter_cols}: {range_cols.reshape(1, -1)[0]}')

    it = 0
    for i in range_rows:
        for j in range_cols:

            progress = it / (len(range_cols) * len(range_rows))

            print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
            print(f"(seed {k}) simulation #{it}, {str(parameter_rows)} = {i}, {str(parameter_cols)} = {j}")
            print(colored(f"{progress * 100:.1f}% of the loop", "blue"))

            """ initialisation """

            # path for the images
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path_run = path_loop + f"simulation_{run_timestamp}/"
            if not os.path.exists(path_run): os.makedirs(path_run)

            # parameters

            seed = k
            grid_size = 200

            V = i
            A = 10
            W = 150
            chi = j

            J = None
            U = None
            dt = None

            # run the simulation
            simulation(
                run_timestamp, path_run, seed, grid_size, # setup
                V, J, U, A, W, chi, dt, eta=2, t_limit=100, # parameters
                adaptive_dt=True, current_scaling=False, power_based_threshold=False, # physics-related options
                cap_from_logs=False, path_uncapped_logs=None, psi_max_fraction=0.5, # option: adjust cap based on previous run
                plot_interval=50, vector_grid=11, # visualization parameters 
                pdf_states=False, i_pdf=(), # saving data for pdf plots
                resistance_plot=True, raw_cluster_plot=True, animation=False, # plot options
                cross_section=False, cross_section_index=None, cross_section_axis='y') # cross-section options

            it += 1
        
    end_time = datetime.now()
    execution_time = end_time - start_time

    print(f"execution time: {execution_time}")
