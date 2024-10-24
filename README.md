# Sustainable waste collection optimization

In this repository we provide our personal implementation of the models described in
[this paper](./docs/ErfanAlirezaSelma-Novel_model_for_sustainable_waste_collection.pdf), by Erfan
Babaee Tirkolaee, Alireza Goli, Selma Gütmen, Gerhard-Wilhelm Weber, Katarzyna Szwedzka.

In particular, two models are implemented:
- *epsilon-constraint method*;
- MOSA-MOIWOA (i.e., *Multi-Objective Simulated Annealing* combined with *Multi-Objective Invasive Weed
Optimization Algorithm*).

Additionally, a comparative study on the two algorithms is provided in [this presentation](./docs/Presentation.pdf).


## Table of contents

- [What you will find in this repository](#what-you-will-find-in-this-repository)
- [Requirements](#requirements)
- [How to prepare a valid dataset](#how-to-prepare-a-valid-dataset)
- [How to use the models](#how-to-use-the-models)
- [References](#references)


## What you will find in this repository

- [`WCO_lib/`](./WCO_lib): library, containing the following modules:
  - [`dataset.py`](./WCO_lib/dataset.py): classes to generate dataset;
  - [`evaluate.py`](./WCO_lib/evaluate.py): functions to compute objectives and evaluation metrics;
  - [`models_exact.py`](./WCO_lib/models_exact.py`): single-objective models for epsilon-constraint solver;
  - [`models_heuristic.py`](./WCO_lib/models_heuristic.py`): various algorithms for MOSA-MOIWOA;
  - [`mutations.py`](./WCO_lib/mutations.py`): functions to perform mutations in MOSA-MOIWOA;
  - [`params.py`](./WCO_lib/params.py`): parameter classes for data and algorithms;
  - [`solve_epsilon.py`](./WCO_lib/solve_epsilon.py`): epsilon-constraint solver;
  - [`solve_moiwoa.py`](./WCO_lib/solve_moiwoa.py`): MOSA-MOIWOA solver;
  - [`subtours.py`](./WCO_lib/subtours.py`): functions to handle subtours in MOSA-MOIWOA;
- [`scripts/`](./scripts): scripts for comparative study;
- [`tests/`](./tests): various tests used during development;
- [`docs/`](./docs): papers and presentations; in particular contains:
  - [`Implementation_details.pdf`](./docs/Implementation_details.pdf): detailed description of the implementations;
  - [`Presentation.pdf`](./docs/Presentation.pdf): presentation of the comparative study;
- [`requirements.txt`](./requirements.txt): list of required packages to run the code.


## Requirements

To use the library you need to have python3 installed on your machine.

The required packages are listed in [`requirements.txt`](./requirements.txt).

**Notice**: to use the epsilon-constraint method you need to have a valid gurobi license. Instructions to obtain
one can be found [here](https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license).

If you want to use MOSA-MOIWOA with a non-default configuration, you also need to create a JSON file containing the
following parameters:
```
"N_0" -> int (default: 10)
"MOSA_T_0" -> float (default: 800)
"MOSA_max_iter" -> int (default: 200)
"MOSA_max_non_improving_iter" -> int (default: 10)
"MOSA_alpha" -> float (default: 0.9)
"MOSA_K": -> float (default: 70)
"MOIWOA_S_min" -> float (default: 9)
"MOIWOA_S_max" -> float (default: 200)
"MOIWOA_N_max" -> int (default: 100)
"MOIWOA_max_iter" -> int (default: 300)
```


## How to prepare a valid dataset


## How to use the models

1. Clone the repository on your machine:
   ```bash
   $ git clone git@github.com:TommasoTarchi/Waste_collection_optimization.git
   ```

2. Place the following lines of code at the beginning of each script using the library:
   ```python
   import sys

   library_path = </path/to/WCO_lib>
   if library_path not in sys.path:
       sys.path.append(library_path)
   ```

3. Before running any model make sure you satisfy all requirements listed in [this section](#requirements) and have
   a valid dataset (please take a look at the [previous section](#how-to-prepare-a-valid-dataset) to correctly generate/prepare
   your dataset).

5. Depending on the model you want to run, import the needed classes and functions:
   - For epsilon-constraint method, here is the minimal import:
     ```python
     from WCO_lib.params import ProblemParams
     from WCO_lib.solve_epsilon import EpsilonSolver
     ```
   - For MOSA-MOIWOA, here is the minimal import:
     ```python
     from WCO_lib.params import ProblemParams, MosaMoiwoaSolverParams
     from WCO_lib.solve_moiwoa import MosaMoiwoaSolver
     ```

6. Create a `ProblemParams` object, specifying the path to the dataset:
   ```python
   problem_params = ProblemParams()
   problem_params.load_from_dir(</path/to/data/directory/>)
   ```

7. Depending on which algorithm you want to use, follow the instructions contained in the following sections.

### Epsilon-constraint method

7. Create an `EpsilonSolver` object passing the problem parameters:
   ```python
   solver = EpsilonSolver(problem_params)
   ```

8. Solve the single objective sub-problems:
   ```python
   solver.solve_single_objectives()
   ```

9. Choose the number of epsilons you want to use <num_epsilon>, then compute the epsilon values and solve the
   multi-objective problem:
   ```python
   solver.compute_epsilon(<num_epsilon>)
   solver.solve_multi_objective()
   ```

10. Retrieve the Pareto solutions:
    ```python
    pareto_solutions = solver.return_pareto_solutions()
    ```
    The returned object is a list of dictionaries, each one representeing a Pareto solution.
    The keys of the dictionaries correspond to the paper's variable names (`x`, `y`, `u`, `LT`, `UT`, `WT`).

### MOSA-MOIWOA

7. Build the graph corresponding to the dataset:
   ```python
   problem_params.build_graph()
   ```

8. Create a `MosaMoiwoaSolverParams` object, specifying the path to parameters file:
   ```python
   solver_params = MosaMoiwoaSolverParams()
   solver_params.load_from_file(</path/to/solver/parameters/file>)
   ```
   **Notice**: if you are ok with default parameters, you can avoid creating the parameters file and skip the
   second line.

10. Create a `MosaMoiwoaSolver` object passing the problem parameters and the solver parameters:
    ```python
    solver = MosaMoiwoaSolver(problem_params, solver_params)
    ```

10. Generate initial solutions:
    ```python
    solver.generate_initial_solutions()
    ```

11. Refine initial solutions using MOSA:
    ```python
    solver.apply_MOSA()
    ```

12. Apply MOIWOA:
    ```python
    solver.apply_MOIWOA()
    ```

13. Retrieve the Pareto solutions:
    ```python
    pareto_solutions = solver.return_pareto_solutions()
    ```
    The returned object is a list of lists, each one representing a Pareto solution (see the
    [implementation details](./docs/Implementation_details.pdf) for complete description).


## References

- Tirkolaee, E.B., Goli, A., Gütmen, S. et al. A novel model for sustainable waste collection arc routing problem:
  Pareto-based algorithms. Ann Oper Res 324, 189–214 (2023).
  [https://doi.org/10.1007/s10479-021-04486-2](https://doi.org/10.1007/s10479-021-04486-2).

- Amine, Khalil, Multiobjective Simulated Annealing: Principles and Algorithm Variants, Advances in Operations
  Research, 2019, 8134674, 13 pages, 2019. [https://doi.org/10.1155/2019/8134674](https://doi.org/10.1155/2019/8134674).
