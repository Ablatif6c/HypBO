"""
This script performs synthetic experiments using the HypBO
optimization algorithm. It takes command line arguments to specify the
function name, dimension, and other parameters. The script runs the HypBO
algorithm for the given function and dimension, and scenarios.
It saves the optimization data in the specified folder path.
"""

import argparse
import os
from multiprocessing import Process
from typing import List

from DBO.bayes_opt.bayesian_optimization import BayesianOptimization
from hypbo import HypBO
from hypothesis import Hypothesis
from utils import get_function, get_scenario_name, get_scenarios

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument(
    "-f",
    "--func_name",
    help="Function name",
    nargs='?',
    type=str,
    const="Branin",
    default="Branin")
parser.add_argument(
    "-d",
    "--dim",
    help="Starting dimension",
    nargs='?',
    type=int,
    const=2,
    default=2)
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
    const=2,
    default=2)
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
    const=10,
    default=10)
parser.add_argument(
    "-bc",
    "--batch",
    help="Batch size",
    nargs='?',
    type=int,
    const=1,
    default=1)
parser.add_argument(
    "-np",
    "--n_processes",
    help="Number of processes",
    nargs='?',
    type=int,
    const=10,
    default=10)
parser.add_argument(
    "-u",
    "--upper_limit",
    help="Upper failure limit",
    nargs='?',
    type=int,
    const=5,
    default=5)
parser.add_argument(
    "-l",
    "--lower_limit",
    help="Lower failure limit",
    nargs='?',
    type=int,
    const=2,
    default=2)
parser.add_argument(
    "-a",
    "--ablation_studies",
    help="Is this an ablation study",
    nargs='?',
    type=bool,
    const=False,
    default=False)

# Read arguments from command line
args = parser.parse_args()
seed_start = args.seed_start
seed_count = args.seed_count
budget = args.budget
n_init = args.n_init
batch = args.batch
n_processes = args.n_processes
func_name = args.func_name
dim = args.dim
ablation_studies = args.ablation_studies
upper_limit = args.upper_limit
lower_limit = args.lower_limit


def run_hypbo(
    func,
    scenarios: List[List[Hypothesis]],
    seed: int = 0,
):
    """
    Run the HypBO optimization algorithm.

    Args:
        func (object): The objective function to be optimized.
        scenarios (List, optional): A list of scenarios, which is a list
        of hypotheses, to be considered during optimization.
        seed (int, optional): The random seed for reproducibility.

    Returns:
        None
    """

    # Get the scenario name
    scenario_name = get_scenario_name(scenario=scenarios)
    print(
        f"--------------- Scenario: {scenario_name}\t Seed: {seed}/{seed_start + seed_count-1}")

    # Params
    pbounds = func.bound
    feature_names = list(pbounds.keys())
    model_kwargs = {
        "pbounds": pbounds,
        "random_seed": seed,
    }

    # Create hypbo with the params and do the search.
    print(f"Upper vs. Lower: {upper_limit} vs. {lower_limit}")
    hypbo = HypBO(
        func=func,
        feature_names=feature_names,
        model=BayesianOptimization,
        model_kwargs=model_kwargs,
        hypotheses=scenarios,
        seed=seed,
        n_processes=n_processes,
        discretization=False,
        global_failure_limit=upper_limit,
        local_failure_limit=lower_limit
    )
    hypbo.search(
        budget=budget,
        n_init=n_init,
        batch=batch,
    )

    # Get the folder path to save the data.
    folder_path = os.path.join(
        "data",
        "HypBO",
    )
    if ablation_studies:
        folder_path = os.path.join(
            folder_path,
            "ablation_studies",
            f"u{upper_limit}_l{lower_limit}")

    # Save the data.
    hypbo.save_data(
        func_name=f"{func.name}_d{func.dim}",
        scenario_name=scenario_name,
        folder_path=folder_path)


def multiprocess_synthetic_experiment(
        func_name: str,
        dim: int = 0):
    """
    Run a multiprocess synthetic experiment.

    Args:
        func_name (str): The name of the function to be processed.
        dim (int, optional): The dimension of the function. Defaults to 0.
    """
    # Print the function name being processed
    print(f"Processing {func_name}...")

    # Get the function based on the given name and dimension
    func = get_function(
        dim=dim,
        name=func_name,
    )

    # Get the last two scenarios for the given function name and dimension
    hypotheses = get_scenarios(
        func_name=func_name,
        dim=dim)

    # Iterate over the range of seeds
    for seed in range(seed_start, seed_start + seed_count):
        processes = []
        # Iterate over the hypotheses
        for hypothesis in hypotheses:
            # Create a new process for each hypothesis
            process = Process(
                target=run_hypbo,
                args=(func, hypothesis, seed),
            )
            # Start the process
            process.start()
            # Add the process to the list
            processes.append(process)

        # Wait for all processes to finish
        for process in processes:
            process.join()


if __name__ == "__main__":
    multiprocess_synthetic_experiment(
        func_name=func_name,
        dim=dim,
    )
