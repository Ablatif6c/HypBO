from experiments.library import get_experiment
from hypbo.hypbo import HypBO
from hypbo.utils import HypothesisHelper
from pathlib import Path
import torch
import argparse

tkwargs = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,
}
# Initialize parser
parser = argparse.ArgumentParser()

# Adding arguments
parser.add_argument(
    "-e",
    "--experiment",
    help="Experiment",
    type=str,
    default="Ackley",
)
parser.add_argument(
    "-d",
    "--dim",
    help="Dimension of the problem",
    type=int,
    default=2,
)
args = parser.parse_args()


if __name__ == "__main__":
    experiment_name = args.experiment
    for random_seed in range(5):
        print(
            f"------------------ Running experiment {experiment_name} with seed {random_seed}"
        )
        experiment = get_experiment(
            experiment_name,
            dim=args.dim,
            random_seed=random_seed,
            noise=False,
        )
        hypbo = HypBO(
            experiment=experiment,
            pbounds=experiment.pbounds,
        )
        optimum = (
            experiment.optimums[0]
            if experiment.optimums
            else torch.zeros(experiment.dim, **tkwargs)
        )
        if not isinstance(optimum, torch.Tensor):
            optimum = torch.tensor(optimum, **tkwargs)
        hypotheses = HypothesisHelper(
            experiment_pbounds=experiment.pbounds,
            size=torch.tensor([2.0] * experiment.dim, **tkwargs),
            optimum=optimum,
        ).create_hypotheses()

        for hypothesis in hypotheses:
            # if hypothesis["name"] == "Good":
            #     continue
            hypbo.add_hypothesis(
                name=hypothesis["name"],
                pbounds=hypothesis["pbounds"],
            )

        hypbo.maximize(
            n_init=5,
            budget=100,
            batch_size=1,
        )
        filepath = (
            Path("data")
            / "new_version"
            / "hypbo"
            / f"{experiment_name}_{experiment.dim}"
            / f"{experiment_name}_s{random_seed}.csv"
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)
        hypbo.save_data(filepath)
