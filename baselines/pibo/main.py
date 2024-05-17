"""
This script runs the piBO algorithm for multiple scenarios and seeds.
It takes command line arguments to specify the function name, dimension,
number of initial samples, optimization budget, starting seed, and number
of seeds.
"""
import argparse
import os
from multiprocessing import Process

from hypermapper import optimizer
from utils import generate_prior, get_function, get_scenarios


def run_pibo(
    func: callable,
    scenario: object,
    n_init: int,
    budget: int,
    seed: int,
    seed_start: int,
    seed_count: int,
) -> None:
    """
    Run the piBO algorithm for a given function and scenario.

    Args:
        func (callable): The function to optimize.
        scenario (object): The scenario to use.
        n_init (int): The number of initial samples.
        budget (int): The optimization budget.
        seed (int): The seed value.
        seed_start (int): The starting seed value.
        seed_count (int): The number of seeds.

    Returns:
        None
    """
    scenario_name = scenario.name
    print(
        f"--------------- Scenario: {scenario_name}\t Seed: {seed}/{seed_start + seed_count-1}")
    # Params
    parameter_ranges = list(func.bound.values())
    application_name = f"{func.name}_{scenario_name}_{seed}"
    prior_ranges = list(zip(*scenario.sol_bounds))

    # Create the prior file.
    priors_folder = "priors"
    os.makedirs(priors_folder, exist_ok=True)
    parameters_file = os.path.join(
        priors_folder,
        f"{application_name}.json")

    # Create output file.
    output_folder = os.path.join(
        "outputs", f"{func.name}_d{func.dim}", scenario_name)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(
        output_folder,
        f"{application_name}_output_samples.csv")
    generate_prior(
        application_name=application_name,
        number_of_initial_samples=n_init,
        optimization_iterations=budget,
        parameter_ranges=parameter_ranges,
        prior_ranges=prior_ranges,
        file_path=parameters_file,
        output_data_file=output_file,
        seed=seed)

    optimizer.optimize(parameters_file, func)
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
    Run the piBO algorithm for multiple scenarios and seeds.

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
    hypotheses = get_scenarios(
        func_name=func_name,
        dim=dim)
    for seed in range(seed_start, seed_start + seed_count):
        processes = []
        for hypothesis in hypotheses:
            process = Process(
                target=run_pibo,
                args=(func, hypothesis, n_init, budget,
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
