import argparse
from multiprocessing import Process
from typing import List

import numpy as np

from bo.bayes_opt.bayesian_optimization import BayesianOptimization
from hbo import HBO
from lamcts.lamcts.MCTS import MCTS
from test_hypotheses import get_function, get_scenarios, get_scenarios_name

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
    "-ds",
    "--dim_start",
    help="Starting dimension",
    nargs='?',
    type=int,
    const=5,
    default=5)
parser.add_argument(
    "-de",
    "--dim_end",
    help="Ending dimension",
    nargs='?',
    type=int,
    const=6,
    default=6)
parser.add_argument(
    "-dstep",
    "--dim_step",
    help="Dimension step",
    nargs='?',
    type=int,
    const=1,
    default=1)
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
    "-b",
    "--budget",
    help="Budget",
    nargs='?',
    type=int,
    const=100,
    default=100)
parser.add_argument(
    "-n",
    "--n_init",
    help="Initialization number",
    nargs='?',
    type=int,
    const=5,
    default=5)
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
    help="Number of rocesses",
    nargs='?',
    type=int,
    const=10,
    default=10)

# Read arguments from command line
args = parser.parse_args()
seed_start = args.seed_start
seed_count = args.seed_count
budget = args.budget
n_init = args.n_init
batch = args.batch
n_processes = args.n_processes
func_name = args.func_name
dim_start = args.dim_start
dim_end = args.dim_end
dim_step = args.dim_step


def run_scenario(
            func,
            scenarios: List = [],
            seed: int = 0,
          ):
    scenario_name = get_scenarios_name(scenarios=scenarios)
    print(f"--------------- Scenario: {scenario_name}\
                \t Seed: {seed}/{seed_start + seed_count-1}")
    # Params
    pbounds = func.bound
    feature_names = list(pbounds.keys())
    model_kwargs = {
                    "pbounds": pbounds,
                    "random_seed": seed,
                    "func": None,
                    }
    hbo = HBO(
        func=func,
        feature_names=feature_names,
        model=MCTS,
        model_kwargs=model_kwargs,
        conjs=scenarios,
        seed=seed,
        n_processes=n_processes,
        discretization=False,
    )
    hbo.search(
        budget=budget,
        n_init=n_init,
        batch=batch,
    )
    hbo.save_data(
        func_name=f"{func.name}_d{func.dim}",
        scenario_name=scenario_name,
        seed=seed,)


def multiprocess_math_experiment(
        func_name: str,
        dim_start: int = 0,
        dim_end: int = 0,
        dim_step: int = 1):
    print(f"Processing {func_name}...")
    for d in range(dim_start, dim_end, dim_step):
        # Get function from name
        func = get_function(
            dim=d,
            name=func_name,
        )

        for seed in range(seed_start, seed_start + seed_count):
            scenarios_list = get_scenarios(
                func_name=func_name,
                dim=d)
            scenarios_chunks = np.array_split(
                scenarios_list,
                3)
            for chunk in scenarios_chunks:
                processes = []
                for scenarios in chunk:
                    process = Process(
                                target=run_scenario,
                                args=(func, scenarios, seed),
                                )
                    process.start()
                    processes.append(process)

                for process in processes:
                    process.join()


if __name__ == "__main__":
    multiprocess_math_experiment(
        func_name=func_name,
        dim_start=dim_start,
        dim_end=dim_end,
        dim_step=dim_step,
    )
