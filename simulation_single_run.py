# 2026.06.04 v1

# corresponds to the above simulation_core version

import numpy as np
import os
import glob
import imageio.v2 as imageio
from datetime import datetime
from simulation_core import simulation

""" initialisation """

# paths for the images and the gif
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path_run = f"simulation_single_run/images_{run_timestamp}/"
if not os.path.exists(path_run): os.makedirs(path_run)
path_gif = path_run + f"/simulation_{run_timestamp}.gif"

# parameters

seed = 42 # randomness control
n = 100 # grid size

dt = 8e-6 # caution: overridden inside the simulation function if adaptive_dt (but that's ok)
J = 10000
U = 100
A = 10 # or 1.0
W = 100
chi = 10

# dt = 8e-6 # caution: overridden inside the simulation function if adaptive_dt (but that's ok)
# J = 2500
# U = 10
# A = 10 # or 1.0
# W = 100
# chi = 10 #10. * 100

# BCs: store _indices_ of nodes where psi is 0 or > 0
hotspots = np.arange(0, n**2, n) # [n**2//2] # potential > 0
coldspots = np.arange(n-1, n**2, n) # potential = 0

# run the simulation
simulation(
    run_timestamp, path_run,
    seed, n, coldspots, hotspots, # setup
    J, U, A, W, chi, dt, eta=2, # parameters
    adaptive_dt=True, current_scaling=True,
    plot_interval=50) 

print("about to make the animation...")

# make the animation
filenames = glob.glob(path_run + 'simulation*.png')
with imageio.get_writer(path_gif, mode='I', fps=10, dpi=200) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("the gif is ready.")