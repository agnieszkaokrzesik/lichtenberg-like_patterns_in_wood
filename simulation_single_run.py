# 2026.06.09 v2

# corresponds to the above simulation_core version

import numpy as np
import os
from datetime import datetime
from simulation_core import simulation

""" initialisation """

# paths for the images and the gif
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path_run = f"simulation_single_run/images_{run_timestamp}/"
if not os.path.exists(path_run): os.makedirs(path_run)

# parameters

seed = 42 # randomness control
n = 100 # grid size

# dt = 8e-6 # caution: overridden inside the simulation function if adaptive_dt (but that's ok) # UWAGA CZEMU TO NIE ROSNIE #TEMP 0609
# J = 1001
# U = 25 # np.inf
# A = 1 # or 1.0
# W = 1000
# chi = 5

dt = 8e-6 # caution: overridden inside the simulation function if adaptive_dt (but that's ok) # UWAGA CZEMU TO NIE ROSNIE #TEMP 0609
J = 10000
U = 200 # np.inf
A = 10 # or 1.0
W = 100
chi = 4

# dt = 8e-6 # caution: overridden inside the simulation function if adaptive_dt (but that's ok)
# J = 2500
# U = 10
# A = 10 # or 1.0
# W = 100
# chi = 10 #10. * 100

# run the simulation
simulation(
    run_timestamp, path_run,
    seed, n, # setup
    J, U, A, W, chi, dt, eta=2, # parameters
    adaptive_dt=True, current_scaling=True,
    cap_from_logs=False, path_uncapped_logs=None,
    plot_interval=50) 