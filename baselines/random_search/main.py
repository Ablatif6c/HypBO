import argparse
import os
from multiprocessing import Process

import numpy as np
from random_search.random_search import RandomOptimizer

from utils import get_function

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument(
    "-f",
    "--func_name",
    help="Function name",
    nargs='?',
    type=str,
    const="HER",
    default="HER")
parser.add_argument(
    "-ds",
    "--dim",
    help="Dimension",
    nargs='?',
    type=int,
    const=9,
    default=9)
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
    "-b",
    "--budget",
    help="Budget",
    nargs='?',
    type=int,
    const=300,
    default=300)
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
    const=5,
    default=5)

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


def run_random(func, seed):
    print(f"--------------- Scenario: No Hypothesis\
                \t Seed: {seed}/{seed_start + seed_count-1}")
    steps = None
    name = f"{func.name}_d{func.dim}"
    constraints = []
    space = func.bound
    if "HER" in func.name:
        steps = [0.25]*10
        steps[5] = 0.2
        name = "HER"
        constraints = [
            '5 - L-Cysteine-50gL - NaCl-3M - NaOH-1M - PVP-1wt - SDS-1wt -'
            ' Sodiumsilicate-1wt - AcidRed871_0gL - '
            'RhodamineB1_0gL - MethyleneB_250mgL']
        space = {
            'AcidRed871_0gL': (0, 5),
            'L-Cysteine-50gL': (0, 5),
            'MethyleneB_250mgL': (0, 5,),
            'NaCl-3M': (0, 5),
            'NaOH-1M': (0, 5),
            'P10-MIX1': (1, 5),
            'PVP-1wt': (0, 5),
            'RhodamineB1_0gL': (0, 5),
            'SDS-1wt': (0, 5),
            'Sodiumsilicate-1wt': (0, 5)}
    random_optimizer = RandomOptimizer(
        space=space,
        steps=steps,
        constraints=constraints,
    )
    random_optimizer.maximize(
        func, init_points=n_init, n_iter=budget, seed=seed)
    folder = os.path.join(
        "data",
        "random_search"
    )

    random_optimizer.save_data(
        func_name=name, folder=folder)


def multiprocess_random_experiment():
    print(f"Processing {func_name}...")
    # Get function from name
    func = get_function(
        dim=dim,
        name=func_name,
    )
    seed_chunks = np.array_split(
        range(seed_start, seed_start + seed_count),
        10)
    for chunk in seed_chunks:
        processes = []
        for seed in chunk:
            seed = int(seed)
            process = Process(
                target=run_random,
                args=(func, seed),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()


if __name__ == "__main__":
    multiprocess_random_experiment()
