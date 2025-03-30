from experiments.library import library
from hypbo.hypbo import HypBO
from hypbo.utils import HypothesisHelper
import torch

tkwargs = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,
}


if __name__ == "__main__":
    experiment = library["ConstrainedGramacy"]
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
        size=torch.tensor([3.0, 3.0], **tkwargs),
        optimum=optimum,
    ).create_hypotheses()

    for hypothesis in hypotheses:
        hypbo.add_hypothesis(
            name=hypothesis["name"],
            pbounds=hypothesis["pbounds"],
        )

    hypbo.maximize(
        n_init=1,
        budget=30,
        batch_size=3,
    )
