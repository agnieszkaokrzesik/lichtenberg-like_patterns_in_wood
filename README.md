# Lichtenberg-like patterns in wood

This is a repository where I keep the code for my master's thesis project - that is, a model simulating the formation of Lichtenberg-like patterns in wet wood under high voltage. 

## Structure

- `preliminary/` - archived preliminary code and results
- `results/` - example results
- `simulation_core.py` - file containing the main simulation function
- `single_run.py` - **script that runs a single simulation**
- `sweep_run.py` - script intended for parameter sweeps

## Requirements, usage & output

Use **Python 3.12**; the required packages are:

    imageio matplotlib numpy scipy termcolor

To run a single simulation of the system, use the `single_run.py` script (`sweep_run.py` is intended for exploration of multiple parameter values). Above the main function call, there is a docstring specifying which parameters to use (some of them are leftovers from previous model variants). Each `single_run.py` run creates a folder `output/single_run/simulation_<timestamp>/` containing an animation of the system (along with the frames), a plot of resistance over time, and other files intended for post-processing.