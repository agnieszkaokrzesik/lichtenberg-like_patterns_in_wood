# 2026.05.28

# created single-run execution of the simulation

import numpy as np
import os
import glob
import imageio.v2 as imageio
from datetime import datetime
from simulation_core import simulation

""" initialisation """

# paths for the images and the gif
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path_images = f"single_run/images_{timestamp}/"
if not os.path.exists(path_images): os.makedirs(path_images)
path_gif = f"single_run/animation_{timestamp}.gif"

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
# J = 5000
# U = 30
# A = 10 # or 1.0
# W = 100
# chi = 8 #10. * 100

# BCs: store _indices_ of nodes where psi is 0 or > 0
hotspots = np.arange(0, n**2, n) # [n**2//2] # potential > 0
coldspots = np.arange(n-1, n**2, n) # potential = 0

# run the simulation
simulation(
    path_images,
    seed, n, coldspots, hotspots, # setup
    J, U, A, W, chi, dt, eta=2) # parameters

# make the animation
filenames = glob.glob(path_images + 'simulation*.png')
with imageio.get_writer(path_gif, mode='I', fps=12, dpi=200) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("the gif is ready.")
