import argparse
from multiprocessing import Process

from bo.bayes_opt.bayesian_optimization import DiscreteBayesianOptimization
from hbo import HBO
from test_hypotheses import get_function, get_scenarios, get_scenarios_name

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument(
    "-s",
    "--seed",
    help="Seed",
    nargs='?',
    type=int,
    const=0,
    default=0)
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
seed = args.seed
budget = args.budget
n_init = args.n_init
batch = args.batch
n_processes = args.n_processes

func = get_function(name="HER")

# Params
pbounds = {
            'AcidRed871_0gL': (0, 5, .25),
            'L-Cysteine-50gL': (0, 5, .25),
            'MethyleneB_250mgL': (0, 5, .25),
            'NaCl-3M': (0, 5, .25),
            'NaOH-1M': (0, 5, .25),
            'P10-MIX1': (1, 5, 0.2,),
            'PVP-1wt': (0, 5, .25),
            'RhodamineB1_0gL': (0, 5, .25),
            'SDS-1wt': (0, 5, .25),
            'Sodiumsilicate-1wt': (0, 5, .25)}
constraints = [
        '5 - L-Cysteine-50gL - NaCl-3M - NaOH-1M - PVP-1wt - SDS-1wt -'
        ' Sodiumsilicate-1wt - AcidRed871_0gL - '
        'RhodamineB1_0gL - MethyleneB_250mgL']
feature_names = list(pbounds.keys())


def run_scenario(
            scenarios=[],
            seed: int = 0,
          ):
    scenario_name = get_scenarios_name(scenarios=scenarios)
    print(f"--------------- Scenario: {scenario_name}")
    model_kwargs = {
                    "pbounds": pbounds,
                    "random_seed": seed,
                    "constraints": constraints,
                    }
    hbo = HBO(
        func=func,
        feature_names=feature_names,
        model=DiscreteBayesianOptimization,
        model_kwargs=model_kwargs,
        conjs=scenarios,
        seed=seed,
        n_processes=n_processes,
    )
    hbo.search(
        budget=budget,
        n_init=n_init,
        batch=batch,
    )
    hbo.save_data(
        func_name=func.name,
        scenario_name=scenario_name,
        seed=seed,)


def multiprocess_her_experiment():
    print(f"Processing HER... seed: {seed}")
    scenarios_list = get_scenarios(func_name="HER")
    processes = []
    for scenarios in scenarios_list[-1:]:
        process = Process(
                    target=run_scenario,
                    args=(scenarios, seed),
                    )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    multiprocess_her_experiment()
