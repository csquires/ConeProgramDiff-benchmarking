# ConeProgramDiff-benchmarking

This repo provides numerical benchmarking results and usage examples for [ConeProgramDiff.jl](https://github.com/tjdiamandis/ConeProgramDiff.jl).

The repo is structured as follows:
* `applications/glasso`: Contains Julia and Python code for the cone program form of the Graphical Lasso algorithm for learning sparse undirected graphs, and for the cross-validation gradient method (CVGM) for hyperparamter optimization.
* `applications/robust_opt`: Contains Julia code for analyzing the sensitivity of a robust optimization problem with respect to its uncertainty sets.
* `benchmarking`: Contains Python and Julia code for generating programs for benchmarking, saving the computation time for solving and computing adjoints/derivatives, and plotting the results.

To create a Python virtual environment with the necessary Python packages, and enter the environment, run:
```
bash setup.sh
source venv/bin/activate
```
