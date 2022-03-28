# Peridynamics

Welcome to the peridynamics research repository of [John Bartlett](https://www.linkedin.com/in/john-bartlett-296a44126/) (PhD Candidate, University of Washington)

[Peridynamics](https://en.wikipedia.org/wiki/Peridynamics) is a nonlocal integral formulation of continuum mechanics. The research shared here focusses on parallelized implementations of the peridynamic method utilizing GPUs.

## GPU-Parallelized Peridynamics

This repository contains two primary GPU implementations of Peridynamics, found in the 'gpupd' folder. The first of these, found in the 'compopt' subdirectory, is compute-optimized, meaning it achives optimal computational performance in large-scale simulations, but is strongly memory-limited. The second, found in the 'memopt' subdirectory, is memory-optimized, which trades some computational performance for an approach that minimizes spatial complexity, allowing extremely large-scale simulation on limited GPU resources. The 'memopt' folder contains both single- and double- precision implementations.

The following python packages are required to run these scripts:
 - math
 - numba
 - numpy
 - os
 - pycuda
 - sys
 - time

Additionally, a Nvidia GPU & driver is necessary to run the scripts on the GPU. If one is not available, the scripts can still be run by enabling gpu-simulation mode, setting the 'ENABLE_CUDA_SIM' enviornment variable to 1 with the line at the top of each script.


## Peridynamic Topology Optimization

The peridynamic topology optimization program can be found in the 'pd_to' folder; it is executed with the 'main.py' script found there. Note that a problem size (nodes per unit length) must be specified; e.g. 'python main.py 100' will run the problem with 100 nodes per unit length.

The following python packages are required for this program:
- numba
- sys
- time

Currently, this program only runs a sample problem (Optimizing a cantilevered beam). Future work will include integration with a geometry modeller to more easily set up different problems to run. 
