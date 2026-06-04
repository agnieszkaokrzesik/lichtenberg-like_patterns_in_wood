# 2026.06.04 v2

# corresponds to the above simulation_core version

import numpy as np
import os
import glob
import imageio.v2 as imageio
from datetime import datetime
from simulation_core import simulation

# path for the images folders
loop_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path_loop = f"simulation_multiple_run/loop_{loop_timestamp}/"
if not os.path.exists(path_loop): os.makedirs(path_loop)

it = 0
for chi in range(1, 10):
    for J in 10.0**np.arange(2, 9): # caution: for J = 1, 10 the system got clogged
                                     # because |I| > 1 (dimensionless threshold) was never satisfied
        it += 1
        print(f"iteration #{it}, chi = {chi}, J = {J}")

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
        A = 10
        W = 100
        #chi = 10

        # BCs: store _indices_ of nodes where psi is 0 or > 0
        hotspots =  np.arange(0, n**2, n) # potential > 0
        coldspots = np.arange(n-1, n**2, n) # potential = 0

        # run the simulation
        simulation(
            run_timestamp, path_run,
            seed, n, # setup
            J, U, A, W, chi, dt, eta=2) # parameters
        
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        