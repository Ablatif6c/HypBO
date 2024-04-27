"""
This script runs the TuRBO algorithm for multiple scenarios and seeds.
It takes command line arguments to specify the function name, dimension,
number of initial samples, optimization budget, starting seed, and number
of seeds.
"""

import argparse
import csv
import multiprocessing
import os
from multiprocessing import Process

import numpy as np
import pandas as pd

from turbo import Turbo1
from utils import get_function


def run_turbo(
    func: callable,
    n_init: int,
    budget: int,
    seed: int,
    seed_start: int,
    seed_count: int,
) -> None:
    """
    Run the TuRBO algorithm for a given function and scenario.

    Args:
        func (callable): The function to optimize.
        n_init (int): The number of initial samples.
        budget (int): The optimization budget.
        seed (int): The seed value.
        seed_start (int): The starting seed value.
        seed_count (int): The number of seeds.

    Returns:
        None
    """
    scenario_name = "No Hypothesis"
    print(
        f"--------------- Scenario: {scenario_name}\t Seed: {seed}/{seed_start + seed_count-1}")

    # Create output file.
    output_folder = os.path.join(
        "outputs__", f"{func.name}_d{func.dim}", scenario_name)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(
        output_folder,
        f"{func_name}_d{func.dim}_{scenario_name}_{seed}_output_samples.csv")

    # Total number of evaluations.
    # TuRBO includes the number of initial samples in the budget.
    # We are only interested in getting the initial samples.
    # That's why we set max_evals = n_init +1
    # to allow for TuRBO to run.
    max_evals = 1 + n_init
    turbo1 = Turbo1(
        f=func,  # Handle to objective function
        lb=func.problem.lb,  # Numpy array specifying lower bounds
        ub=func.problem.ub,  # Numpy array specifying upper bounds
        n_init=n_init,  # Number of initial bounds from an Latin hypercube
        max_evals=max_evals,  # Maximum number of evaluations
        verbose=True,  # Print information from each batch
        seed=seed,  # Random seed
    )
    turbo1.optimize()

    # Read the existing data without init samples.
    data = pd.read_csv(output_file)
    init_data = pd.DataFrame(columns=data.columns)
    # Only append the init samples to the data frame before the data.
    for i in range(n_init):
        sample, value = turbo1.X[i], turbo1.fX[i]
        new_row = list(sample) + list(value)
        init_data.loc[len(init_data)] = new_row  # Append new_row to init_data

    # Append the init samples to the data frame before the data.
    data = pd.concat([init_data, data], ignore_index=True)
    # Write the data to the output file.
    data.to_csv(output_file)

    print(f"--------------- End scenario: {scenario_name}\
                \t Seed: {seed}/{seed_start + seed_count-1}")


def multiprocess_synthetic_experiment(
        func_name: str,
        dim: int,
        n_init: int,
        budget: int,
        seed_start: int,
        seed_count: int) -> None:
    """
    Run the TuRBO algorithm for multiple scenarios and seeds.

    Args:
        func_name (str): The name of the function to optimize.
        dim (int): The dimension of the function.
        n_init (int): The number of initial samples.
        budget (int): The optimization budget.
        seed_start (int): The starting seed value.
        seed_count (int): The number of seeds.

    Returns:
        None
    """
    print(f"Processing {func_name}...")

    func = get_function(
        dim=dim,
        name=func_name,
    )
    num_cpus = multiprocessing.cpu_count()
    seeds = range(seed_start, seed_start + seed_count)
    seed_chunks = np.array_split(seeds, num_cpus)

    for seed_chunk in seed_chunks:
        processes = []
        for seed in seed_chunk:
            seed = int(seed)
            process = Process(
                target=run_turbo,
                args=(func, n_init, budget,
                      seed, seed_start, seed_count),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional arguments
    parser.add_argument(
        "-f",
        "--func_name",
        help="Function name",
        nargs='?',
        type=str,
        const="Ackley",
        default="Ackley")
    parser.add_argument(
        "-d",
        "--dim",
        help="Starting dimension",
        nargs='?',
        type=int,
        const=5,
        default=5)
    parser.add_argument(
        "-ss",
        "--seed_start",
        help="Starting seed",
        nargs='?',
        type=int,
        const=0,
        default=0)
    parser.add_argument(
        "-sc",
        "--seed_count",
        help="Seed count",
        nargs='?',
        type=int,
        const=50,
        default=50)
    parser.add_argument(
        "-n",
        "--n_init",
        help="Initialization number",
        nargs='?',
        type=int,
        const=5,
        default=5)
    parser.add_argument(
        "-b",
        "--budget",
        help="Budget",
        nargs='?',
        type=int,
        const=100,
        default=100)

    # Read arguments from command line
    args = parser.parse_args()
    seed_start = args.seed_start
    seed_count = args.seed_count
    budget = args.budget
    n_init = args.n_init
    func_name = args.func_name
    dim = args.dim

    multiprocess_synthetic_experiment(
        func_name=func_name,
        dim=dim,
        n_init=n_init,
        budget=budget,
        seed_start=seed_start,
        seed_count=seed_count
    )
