# 2026.05.28

# created multiple-run execution of the simulation

import numpy as np
import os
import glob
import imageio.v2 as imageio
from datetime import datetime
from simulation_core import simulation

# path for the images folders
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path_parent = f"multiple_run/loop_{timestamp}/"
if not os.path.exists(path_parent): os.makedirs(path_parent)

it = 0
for chi in 10.0**np.arange(0, 11):
    for J in 10.0**np.arange(0, 11): # caution: for J = 1, 10 the system got clogged
                                     # because |I| > 1 (dimensionless threshold) was never satisfied
        it += 1
        print(f"it #{it}, chi & J:", chi, J)

        """ initialisation """

        # path for the images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_images = path_parent + f"images_{timestamp}/"
        if not os.path.exists(path_images): os.makedirs(path_images)

        # parameters

        seed = 42 # randomness control
        n = 100 # grid size

        dt = 8e-6 # caution: overridden inside the simulation function if adaptive_dt (but that's ok)
        #J = 2500
        U = J / 100 # or np.inf
        A = 10
        W = 100
        #chi = 10

        # BCs: store _indices_ of nodes where psi is 0 or > 0
        hotspots =  np.arange(0, n**2, n) # potential > 0
        coldspots = np.arange(n-1, n**2, n) # potential = 0

        # run the simulation
        simulation(
            path_images,
            seed, n, coldspots, hotspots, # setup
            J, U, A, W, chi, dt, eta=2) # parameters
        
