# bgc-sbo

Marine ecosystem models are important to identify the processes that affects for example the global carbon cycle. Computation of an annual periodic solution (i.e., a steady annual cycle) for these models requires a high computational effort.

The parameter identification is a challenging task for marine ecosystem models. Therefore, we implemented a surrogate-based optimization (SBO) to identify optimal model parameters for marine ecosystem models (see Python package sbo and scripts SurrogateBasedOptimization).



## Installation

To clone this project with **git** run:
>git clone https://github.com/slawig/bgc-sbo.git



## Usage

The project consists of the Python package sbo and Python scripts in the directory SurrogateBasedOptimization to start a optimization.



### Python package sbo

This package summarizes the functions to identify the model parameters of a marine ecosystem model using the SBO.

This package contains three subpackages:
- database:
  Consists of functions to store the configuation data and results of a parameter identification in a database and read them out again.
- plot:
  Consists of functions to plot the results of a parameter identification.
- sbo:
  Implementation of the surrogate-based optimization.


### Python scripts

Python scripts exist for the application to start the simulations and evaluate them.

The scripts for a parameter identification using the surrogate-based optimization are available in the directory `SurrogateBasedOptimization`:
* The script `SBO_Config.py` contains additional configurations for some parameter identification runs.
* The script `SBO_StartJob.py` starts the optimization runs using the configurations defined in the database and the file `SBO_Config.py`.
* The script `SBO_StartPlot.py` visualizes the parameter identification process.



## License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
