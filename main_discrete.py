"""
This script performs the replica of the Photocatalytic Hydrogen Production
experiment using the HypBO optimization algorithm. It takes command line
arguments to specify the budget, initial samples count, and other parameters.
The script runs the HypBO algorithm for the defined hypotheses in the utils
file:
- perfect_hindsight
- what_we_kew
- bizarro_world
- virtual_chemists_team

It saves the optimization data in the specified folder path.
"""

import argparse
from multiprocessing import Process

from DBO.bayes_opt.bayesian_optimization import DiscreteBayesianOptimization
from hypbo import HypBO
from utils import get_function, get_scenario_name, get_scenarios

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument(
    "-s", "--seed", help="Seed", nargs="?", type=int, const=0, default=0
)
parser.add_argument(
    "-b", "--budget", help="Budget", nargs="?", type=int, const=10, default=10
)
parser.add_argument(
    "-n",
    "--n_init",
    help="Initialization number",
    nargs="?",
    type=int,
    const=5,
    default=5,
)
parser.add_argument(
    "-bc",
    "--batch",
    help="Batch size",
    nargs="?",
    type=int,
    const=1,
    default=1,
)
parser.add_argument(
    "-np",
    "--n_processes",
    help="Number of processes",
    nargs="?",
    type=int,
    const=5,
    default=5,
)

# Read arguments from command line
args = parser.parse_args()
seed = args.seed
budget = args.budget
n_init = args.n_init
batch = args.batch
n_processes = args.n_processes

# Getting the Photocatalytic Hydrogen Production function.
func = get_function(name="HER")

# Params
pbounds = {
    "AcidRed871_0gL": (0, 5, 0.25),
    "L-Cysteine-50gL": (0, 5, 0.25),
    "MethyleneB_250mgL": (0, 5, 0.25),
    "NaCl-3M": (0, 5, 0.25),
    "NaOH-1M": (0, 5, 0.25),
    "P10-MIX1": (
        1,
        5,
        0.2,
    ),
    "PVP-1wt": (0, 5, 0.25),
    "RhodamineB1_0gL": (0, 5, 0.25),
    "SDS-1wt": (0, 5, 0.25),
    "Sodiumsilicate-1wt": (0, 5, 0.25),
}
constraints = [
    "5 - L-Cysteine-50gL - NaCl-3M - NaOH-1M - PVP-1wt - SDS-1wt -"
    " Sodiumsilicate-1wt - AcidRed871_0gL - "
    "RhodamineB1_0gL - MethyleneB_250mgL"
]
feature_names = list(pbounds.keys())


def run_scenario(scenario=[], seed: int = 0):
    """
    Run a single scenario of the Photocatalytic Hydrogen Production experiment.

    Args:
        scenarios (list): List of scenarios to run.
        seed (int): Seed value for random number generation.

    """
    # Get the name of the scenario
    scenario_name = get_scenario_name(scenario=scenario)
    print(f"--------------- Scenario: {scenario_name}")

    # Set the model kwargs
    model_kwargs = {
        "pbounds": pbounds,
        "random_seed": seed,
        "constraints": constraints,
    }

    # Initialize HypBO object
    hbo = HypBO(
        func=func,
        feature_names=feature_names,
        model=DiscreteBayesianOptimization,
        model_kwargs=model_kwargs,
        hypotheses=scenario,
        seed=seed,
        n_processes=n_processes,
    )

    # Perform the search
    hbo.search(budget=budget, n_init=n_init, batch=batch)

    # Save the data
    hbo.save_data(
        func_name=func.name,
        scenario_name=scenario_name,
        seed=seed,
    )


def multiprocess_her_experiment():
    """
    Run the Photocatalytic Hydrogen Production experiment.
    """
    # Print the function name being processed
    print(f"Processing HER... seed: {seed}")

    # Get the scenarios
    scenarios_list = get_scenarios(func_name="HER")

    # Run the scenarios in parallel
    processes = []
    for scenarios in scenarios_list:
        process = Process(target=run_scenario, args=(scenarios, seed))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    multiprocess_her_experiment()
