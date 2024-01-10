import argparse
import csv
import os
import warnings
from multiprocessing import Process

from lamcts import MCTS
from utils import get_function, get_scenario_name, get_scenarios

warnings.filterwarnings("ignore")


def run_lamcts_init_hyp(
    func,
    scenario,
    n_init: int,
    budget: int,
    seed: int,
    seed_start: int,
    seed_count: int,
):
    """
    Run the LAMCTS algorithm for a given function and scenario.

    Args:
        func: The function to optimize.
        scenario: The scenario to use.
        seed (int): The seed value.
        seed_start (int): The starting seed value.
        seed_count (int): The number of seeds.

    Returns:
        None
    """
    scenario_name = get_scenario_name(scenario)
    print(
        f"--------------- Scenario: {scenario_name}\t Seed: {seed}/{seed_start + seed_count-1}")
    # Params
    param_space = func.bound

    # Get the initial space.
    init_spaces = []
    for hypothesis in scenario:
        values = list(zip(hypothesis.sol_bounds[0], hypothesis.sol_bounds[1]))
        init_space = {key: value for key,
                      value in zip(param_space.keys(), values)}
        init_spaces.append(init_space)

    # Create output file.
    output_folder = os.path.join(
        "outputs", f"{func.name}_d{func.dim}", scenario_name)
    os.makedirs(output_folder, exist_ok=True)
    csv_filepath = os.path.join(
        output_folder,
        f"{func.name}_{scenario_name}_{seed}_output_samples.csv"
    )

    agent = MCTS(
        lb=func.problem.lb,              # the lower bound of each problem dimensions
        ub=func.problem.ub,              # the upper bound of each problem dimensions
        dims=func.dim,          # the problem dimensions
        ninits=n_init,      # the number of random samples used in initializations
        func=func,               # function object to be optimized
        Cp=1,              # Cp for MCTS
        leaf_size=10,  # tree leaf size
        kernel_type="rbf",  # SVM configruation
        gamma_type="auto",  # SVM configruation
        init_spaces=init_spaces,  # initial space
    )

    agent.search(iterations=budget)

    # Save the samples and their values in a CSV file
    with open(csv_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write column headers
        writer.writerow(list(param_space.keys()) + ["Value"])
        for sample, value in agent.samples:
            # Write sample and its value
            writer.writerow(list(sample) + [-value])

    print(f"--------------- End scenario: {scenario_name}\
                \t Seed: {seed}/{seed_start + seed_count-1}")


def multiprocess_synthetic_experiment(
        func_name: str,
        dim: int,
        n_init: int,
        budget: int,
        seed_start: int,
        seed_count: int):
    """
    Run the LAMCTS algorithm (initialized in the hypothesis subspace) for multiple scenarios and seeds.

    Args:
        func_name (str): The name of the function to optimize.
        dim (int): The dimension of the function.
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
    scenarios = get_scenarios(
        func_name=func_name,
        dim=dim)
    for seed in range(seed_start, seed_start + seed_count):
        processes = []
        for scenario in scenarios:
            process = Process(
                target=run_lamcts_init_hyp,
                args=(func, scenario, n_init, budget,
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
        const=1,
        default=1)
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
        const=2,
        default=2)

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
