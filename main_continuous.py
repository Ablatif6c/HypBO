import argparse
import os
from multiprocessing import Process
from typing import List

from DBO.bayes_opt.bayesian_optimization import BayesianOptimization
from hypbo import HypBO
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
    const=5,
    default=5)
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
    const=True,
    default=True)

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
    scenarios: List = [],
    seed: int = 0,
):
    scenario_name = get_scenario_name(scenario=scenarios)
    print(f"--------------- Scenario: {scenario_name}\
                \t Seed: {seed}/{seed_start + seed_count-1}")
    # Params
    pbounds = func.bound
    feature_names = list(pbounds.keys())
    model_kwargs = {
        "pbounds": pbounds,
        "random_seed": seed,
        # "func": None,
    }

    # Create hypbo with the params and do the search.
    print(f"Global vs. Local: {upper_limit} vs. {lower_limit}")
    hypbo = HypBO(
        func=func,
        feature_names=feature_names,
        model=BayesianOptimization,
        model_kwargs=model_kwargs,
        hypotheses=scenarios,
        seed=seed,
        n_processes=n_processes,
        discretization=False,
        global_failire_limit=upper_limit,
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
        seed=seed,
        folder_path=folder_path)


def multiprocess_synthetic_experiment(
        func_name: str,
        dim: int = 0):
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
                target=run_hypbo,
                args=(func, hypothesis, seed),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()


if __name__ == "__main__":
    multiprocess_synthetic_experiment(
        func_name=func_name,
        dim=dim,
    )
