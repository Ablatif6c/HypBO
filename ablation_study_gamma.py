"""
This script performs a gamma ablation study using the HypBO optimization
algorithm. It explores different gamma values, synthetic functions, and
scenarios to evaluate HypBO's performance. The results are saved in separate
folders based on the gamma value.

The main function `run_hypbo` runs the HypBO optimization algorithm for a
given set of parameters. It takes the gamma value, function name, dimension,
scenario, trial number, total number of trials, batch size, number of initial
points, and optimization budget as input.

The script runs the ablation study in parallel using concurrent.futures.
Each combination of gamma value, function, dimension, and scenario is
processed in a separate process.

"""

import concurrent.futures
import os
from typing import List

import numpy as np

from DBO.bayes_opt.bayesian_optimization import BayesianOptimization
from hypbo import HypBO
from hypothesis import Hypothesis
from utils import get_function, get_scenario_name, get_scenarios


def run_hypbo(gamma: float, function_name: str, dimension: int,
              scenario: List[Hypothesis], trial: int,
              trials: int, batch: int, n_init: int, budget: int) -> None:
    """
    Run the HypBO optimization algorithm.

    Args:
        gamma (float): The gamma percentage controlling the improvement.
        function_name (str): The name of the function to optimize.
        dimension (int): The dimension of the function.
        scenario (List[Hypothesis]): The list of hypotheses.
        trial (int): The current trial number.
        trials (int): The total number of trials.
        batch (int): The batch size.
        n_init (int): The number of initial points.
        budget (int): The optimization budget.

    Returns:
        None
    """
    print(f"gamma: {gamma}")
    print(f"\tfunction: {function_name}, dimension: {dimension}")
    print(f"\t\tscenario: {get_scenario_name(scenario=scenario)}")
    print(f"\t\t\ttrial: {trial+1}/{trials}")

    # Get the function to optimize
    func = get_function(function_name, dimension)

    # HypBO parameters
    param_bounds = func.bound
    feature_names = list(param_bounds.keys())
    model_kwargs = {
        "pbounds": param_bounds,
        "random_seed": trial
    }
    hypbo_params = {
        'func': func,
        'feature_names': feature_names,
        'model': BayesianOptimization,
        'model_kwargs': model_kwargs,
        'hypotheses': scenario,
        'seed': trial,
        'discretization': False,
        'global_failure_limit': 5,
        'local_failure_limit': 2,
        'gamma': gamma
    }

    # Create the HypBO object
    hypbo = HypBO(**hypbo_params)

    # Run the optimization
    hypbo.search(budget=budget, n_init=n_init, batch=batch)

    # Save the data.
    # Get the folder path to save the data.
    folder_path = os.path.join(
        "data", "ablation_studies",  "gamma", str(gamma))

    # Create folders if they don't exist
    os.makedirs(folder_path, exist_ok=True)

    hypbo.save_data(
        func_name=f"{func.name}_d{func.dim}",
        scenario_name=get_scenario_name(scenario=scenario),
        folder_path=folder_path)


if __name__ == '__main__':
    # Generating 5 logarithmically spaced values between 0.1 and 1
    gamma_values = np.logspace(-1, 0, 5)

    # Define the synthetic functions for testing
    synthetic_functions = [
        ("Branin", 2),
        ("Sphere", 2),
        ("Ackley", 5),
    ]

    # Define the parameters for the ablation study
    trials = 20
    budget = 50
    n_init = 5
    batch = 1

    # Run the ablation study in parallel
    processes = []
    for trial in range(trials):
        for gamma in gamma_values:
            for function_name, dimension in synthetic_functions:
                all_scenarios = get_scenarios(function_name, dimension)[:3]
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = []
                    for scenario in all_scenarios:
                        future = executor.submit(
                            run_hypbo, gamma, function_name, dimension,
                            scenario, trial, trials, batch, n_init, budget)
                        futures.append(future)
                    concurrent.futures.wait(futures)
